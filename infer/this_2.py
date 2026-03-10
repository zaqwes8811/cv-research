#!/usr/bin/env python3
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams


def nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate detections
    
    Args:
        detections: list of detection dictionaries with 'bbox' and 'score' keys
        iou_threshold: IoU threshold for suppression
    
    Returns:
        list: filtered detections after NMS
    """
    if len(detections) == 0:
        return []
    
    # Convert detections to format needed for NMS
    boxes = []
    scores = []
    class_ids = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        boxes.append([x1, y1, x2, y2])
        scores.append(det['score'])
        class_ids.append(det['class_id'])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Apply NMS
    selected_indices = []
    
    # Process each class separately
    unique_classes = np.unique(class_ids)
    
    for class_id in unique_classes:
        # Get indices for current class
        class_indices = np.where(np.array(class_ids) == class_id)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Get boxes and scores for current class
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        
        # Sort by score descending
        sorted_indices = np.argsort(class_scores)[::-1]
        
        # Apply NMS for this class
        while len(sorted_indices) > 0:
            # Keep the detection with highest score
            current_idx = sorted_indices[0]
            selected_indices.append(class_indices[current_idx])
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining detections
            current_box = class_boxes[current_idx]
            remaining_boxes = class_boxes[sorted_indices[1:]]
            
            # Calculate IoU
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            
            union = area_current + area_remaining - intersection
            iou = intersection / np.maximum(union, 1e-6)
            
            # Keep detections with IoU less than threshold
            keep_indices = np.where(iou <= iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
    
    # Return filtered detections
    selected_indices = sorted(selected_indices)
    return [detections[i] for i in selected_indices]

def parse_yolov8n_output(output_tensor, conf_threshold=0.5, img_width=640, img_height=640):
    """
    Парсинг выходного тензора YOLOv8n формата [1, 80, N, 5]
    
    Args:
        output_tensor: numpy array формы [1, 80, N, 5]
        conf_threshold: порог уверенности (0.0 - 1.0)
        img_width, img_height: размеры входного изображения
    
    Returns:
        list: список словарей с детекциями
    """
    
    # Убираем batch dimension
    output_tensor = np.array(output_tensor)

    tensor = output_tensor[0]  # теперь [80, N, 5]
    num_classes, num_proposals = output_tensor.shape
    print(tensor)
    
    print(f"Обработка тензора: {num_classes} классов, {num_proposals} предложений на класс")
    
    detections = []
    
    # Проходим по всем классам
    for class_id in range(num_classes):
        # Проходим по всем предложениям для этого класса
        for proposal_idx in range(num_proposals):
            # Извлекаем координаты и уверенность
            x1 = tensor[class_id, proposal_idx, 0]
            y1 = tensor[class_id, proposal_idx, 1]
            x2 = tensor[class_id, proposal_idx, 2]
            y2 = tensor[class_id, proposal_idx, 3]
            confidence = tensor[class_id, proposal_idx, 4]
            
            # Фильтруем по уверенности
            if confidence > conf_threshold:
                # Конвертируем в int для отрисовки
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': get_coco_class_name(class_id)  # опционально
                })
    
    print(f"Найдено детекций: {len(detections)}")
    return detections

class YOLOv8nHailo:
    def __init__(self, hef_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        # COCO dataset class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Load HEF model
        print(f"Loading model: {hef_path}")
        self.hef = HEF(hef_path)
        
        # Get model info
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_info = self.hef.get_output_vstream_infos()[0]
        
        print(f"Model loaded:")
        print(f"  Input shape: {self.input_info.shape}")
        print(f"  Output shape: {self.output_info.shape}")
        print(f"  Input name: {self.input_info.name}")
        print(f"  Output name: {self.output_info.name}")
        
        self.input_height, self.input_width = self.input_info.shape[:2]
    
    def preprocess(self, image_path):
        """Preprocess image for YOLOv8"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = img.copy()
        self.original_height, self.original_width = img.shape[:2]
        
        print(f"Original image size: {self.original_width} x {self.original_height}")
        
        # Convert BGR to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # Convert HWC to CHW and add batch dimension
        input_tensor = np.transpose(resized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
        
        return input_tensor

    def parse_output_3x3_grid(self, output):
        """Парсинг для выходной сетки 3x3"""
        detections = []
        
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]
            
            grid_size = 9
            cell_size = self.original_width / grid_size
            
            for class_id, class_dets in enumerate(batch_output):
                if isinstance(class_dets, np.ndarray) and class_dets.size > 0:
                    for det in class_dets:
                        if len(det) >= 5:
                            # Координаты в системе сетки 3x3
                            x1_grid, y1_grid, x2_grid, y2_grid, conf = det[:5]
                            
                            if conf > self.conf_threshold:
                                # Конвертация в пиксели
                                # Сначала определяем, в каких ячейках находятся углы
                                x1_cell = int(x1_grid * grid_size)
                                y1_cell = int(y1_grid * grid_size)
                                x2_cell = int(x2_grid * grid_size)
                                y2_cell = int(y2_grid * grid_size)
                                
                                # Смещения внутри ячеек
                                x1_offset = (x1_grid * grid_size) - x1_cell
                                y1_offset = (y1_grid * grid_size) - y1_cell
                                x2_offset = (x2_grid * grid_size) - x2_cell
                                y2_offset = (y2_grid * grid_size) - y2_cell
                                
                                # Позиции в пикселях
                                x1 = int((x1_cell + x1_offset) * cell_size)
                                y1 = int((y1_cell + y1_offset) * cell_size)
                                x2 = int((x2_cell + x2_offset) * cell_size)
                                y2 = int((y2_cell + y2_offset) * cell_size)
                                
                                # Проверка границ
                                x1 = max(0, min(x1, self.original_width))
                                y1 = max(0, min(y1, self.original_height))
                                x2 = max(0, min(x2, self.original_width))
                                y2 = max(0, min(y2, self.original_height))
                                
                                if x2 > x1 and y2 > y1:
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'score': conf,
                                        'class_id': class_id,
                                        'class_name': self.class_names[class_id]
                                    })
        
        return detections

    def parse_output_grid(self, output):
        """Парсинг с учетом сетки feature map"""
        detections = []
        
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]
            
            # Определяем размер сетки из данных
            # Судя по паттерну, это 9x9 или 10x10
            grid_size = 3  # или определите автоматически
            
            for class_id, class_dets in enumerate(batch_output):
                if isinstance(class_dets, np.ndarray) and class_dets.size > 0:
                    for det in class_dets:
                        if len(det) >= 5:
                            # Ваши координаты - это индексы в сетке!
                            y1_grid, x1_grid, y2_grid, x2_grid, conf = det[:5]
                            
                            if conf > self.conf_threshold:
                                # Конвертируем из координат сетки в пиксели
                                cell_size = self.original_width / grid_size
                                
                                x1 = int(x1_grid * grid_size * cell_size)
                                y1 = int(y1_grid * grid_size * cell_size)
                                x2 = int(x2_grid * grid_size * cell_size)
                                y2 = int(y2_grid * grid_size * cell_size)
                                
                                # Или проще: x1 = int(x1_grid * self.original_width)
                                # Потому что x1_grid уже в диапазоне 0-1, но это ИНДЕКС ЯЧЕЙКИ!
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'score': conf,
                                    'class_id': class_id,
                                    'class_name': self.class_names[class_id]
                                })
        
        return detections
    
    def parse_output(self, output):
        """Parse the model output with proper coordinate scaling"""
        detections = []
        
        # Output is list with 1 batch element
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]  # List of 80 class outputs
            
            for class_id, class_output in enumerate(batch_output):
                if class_id >= len(self.class_names):
                    continue
                
                # class_output is a numpy array of shape [num_detections, 5]
                if isinstance(class_output, np.ndarray) and class_output.size > 0:
                    print(f"Class {class_id} ({self.class_names[class_id]}): {class_output.shape} detections")
                    
                    for det_idx, det in enumerate(class_output):
                        if len(det) >= 5:
                            # Values are normalized [0-1] coordinates in [x1, y1, x2, y2, score] format
                            x1_norm = float(det[1]) * 3
                            y1_norm = float(det[0]) * 3
                            x2_norm = float(det[3]) * 3
                            y2_norm = float(det[2]) * 3
                            score = float(det[4])
                            
                            # if det_idx == 0 and class_id == 0:
                            print(f"  Normalized coordinates: x1={x1_norm:.3f}, y1={y1_norm:.3f}, x2={x2_norm:.3f}, y2={y2_norm:.3f}, score={score:.3f}")
                            
                            # if score > self.conf_threshold:
                            # Scale normalized coordinates to original image size
                            x1 = int(x1_norm * self.original_width) % self.original_width
                            y1 = int(y1_norm * self.original_height) % self.original_height
                            x2 = int(x2_norm * self.original_width) % self.original_width
                            y2 = int(y2_norm * self.original_height) % self.original_height
                            
                            # Ensure coordinates are within image bounds
                            x1 = max(0, min(x1, self.original_width))
                            y1 = max(0, min(y1, self.original_height))
                            x2 = max(0, min(x2, self.original_width))
                            y2 = max(0, min(y2, self.original_height))
                            
                            # Only add if box has positive area
                            # if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': score,
                                'class_id': class_id,
                                'class_name': self.class_names[class_id]
                            })
                            #if len(detections) <= 5:  # Print first few valid detections
                            print(f"  ✓ Valid box: ({x1}, {y1}) -> ({x2}, {y2}) area={ (x2-x1)*(y2-y1) }")
    
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        print(f"\nDrawing {len(detections)} detections, shape: {image.shape}")
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_name = det['class_name']
            
            # Generate color based on class
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1-20), (x1+w, y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # if i == 6:
            #     break
        
        return image
    
    def run(self, image_path, output_path="result.jpg"):
        """Run inference on image"""
        # Preprocess
        input_tensor = self.preprocess(image_path)
        
        # Run inference
        print(f"\nRunning inference with confidence threshold: {self.conf_threshold}")
        with VDevice() as device:
            params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(self.hef, params)[0]
            
            input_params = InputVStreamParams.make_from_network_group(network_group)
            output_params = OutputVStreamParams.make_from_network_group(network_group)
            
            with network_group.activate(), \
                 InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                
                results = infer_pipeline.infer({self.input_info.name: input_tensor})

                print(results)
                output = results[self.output_info.name]
        
        # Parse detections
        # print("detections shape: ")
        # for v in output:
        #     for k in v:
        #         print("v:", len(output), len(v), len(k))
        #         print(k)

        #parse_yolov8n_output(output)
        detections = nms(self.parse_output(output))

        
        print(f"\nFound {len(detections)} valid objects:")
        
        # Group by class
        from collections import Counter
        class_counts = Counter([d['class_name'] for d in detections])
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
        
        # Draw and save
        if detections:
            result_image = self.draw_detections(self.original_image.copy(), detections)
            cv2.imwrite(output_path, result_image)
            print(f"\nResult saved to: {output_path}")
            
            # Also save with detections overlaid
            cv2.imwrite("detections_only.jpg", result_image)
        else:
            print("\nNo valid boxes to draw")
            # Save original image for reference
            cv2.imwrite(output_path, self.original_image)
        
        return detections

def main():
    import os
    
    # Configuration
    hef_path = "yolov8n.hef"
    image_path = "output_640x640.jpg"
    # image_path = "original.jpg"
    image_path = "wine_glass_ubc.jpg"
    image_path = "images (1).jpeg"
    image_path = "test_image.jpg"
    
    if not os.path.exists(hef_path):
        print(f"Model not found: {hef_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test with different thresholds
    thresholds = [0.3]
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with confidence threshold: {thresh}")
        print('='*60)
        
        yolo = YOLOv8nHailo(hef_path, conf_threshold=thresh)
        detections = yolo.run(image_path, f"result_thresh_{thresh}.jpg")
        
        if detections:
            print(f"\n✓ Found {len(detections)} valid objects with threshold {thresh}")
            #break
        else:
            print(f"✗ No valid objects found with threshold {thresh}")

if __name__ == "__main__":
    main()