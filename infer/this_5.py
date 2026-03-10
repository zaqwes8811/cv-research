#!/usr/bin/env python3
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams

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
        
        self.input_height, self.input_width = self.input_info.shape[:2]
        print(f"Model loaded: {self.input_width} x {self.input_height}")
    
    def preprocess(self, image_path):
        """Preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = img.copy()
        self.orig_h, self.orig_w = img.shape[:2]
        
        print(f"Original image: {self.orig_w} x {self.orig_h}")
        
        # Resize to model input size
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # Convert to CHW format
        input_tensor = np.transpose(resized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
        
        return input_tensor
    
    def parse_output(self, output):
        """Parse YOLOv8 output - format: [x_center, y_center, width, height, class0_conf]"""
        detections = []
        
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]  # List of 80 class outputs
            
            for class_id, class_output in enumerate(batch_output):
                if class_id >= len(self.class_names):
                    continue
                
                if isinstance(class_output, np.ndarray) and class_output.size > 0:
                    # class_output is array of shape [num_detections, 5]
                    for det in class_output:
                        if len(det) >= 5:
                            # Extract values - all normalized 0-1
                            x_center_norm = float(det[0])
                            y_center_norm = float(det[1])
                            width_norm = float(det[2])
                            height_norm = float(det[3])
                            confidence = float(det[4])
                            
                            if confidence > self.conf_threshold:
                                # Clip to valid range
                                x_center_norm = max(0, min(x_center_norm, 1.0))
                                y_center_norm = max(0, min(y_center_norm, 1.0))
                                width_norm = max(0, min(width_norm, 1.0))
                                height_norm = max(0, min(height_norm, 1.0))
                                
                                # Convert center format to corner format
                                x1_norm = x_center_norm - width_norm / 2
                                y1_norm = y_center_norm - height_norm / 2
                                x2_norm = x_center_norm + width_norm / 2
                                y2_norm = y_center_norm + height_norm / 2
                                
                                # Convert to original image pixel coordinates
                                x1 = int(x1_norm * self.orig_w)
                                y1 = int(y1_norm * self.orig_h)
                                x2 = int(x2_norm * self.orig_w)
                                y2 = int(y2_norm * self.orig_h)
                                
                                # Ensure x1 < x2 and y1 < y2
                                if x1 > x2:
                                    x1, x2 = x2, x1
                                if y1 > y2:
                                    y1, y2 = y2, y1
                                
                                # Clip to image boundaries
                                x1 = max(0, min(x1, self.orig_w))
                                y1 = max(0, min(y1, self.orig_h))
                                x2 = max(0, min(x2, self.orig_w))
                                y2 = max(0, min(y2, self.orig_h))
                                
                                # Calculate area
                                area = (x2 - x1) * (y2 - y1)
                                min_area = (self.orig_w * self.orig_h) * 0.005  # 0.5% of image
                                
                                # Only keep reasonably sized boxes
                                if area > min_area:
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'score': confidence,
                                        'class_id': class_id,
                                        'class_name': self.class_names[class_id],
                                        'area': area
                                    })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply Non-Maximum Suppression
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections, iou_threshold=0.5):
        """Remove overlapping boxes"""
        if len(detections) <= 1:
            return detections
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove boxes with high IoU with best
            detections = [d for d in detections 
                         if self.compute_iou(best['bbox'], d['bbox']) < iou_threshold]
        
        return keep
    
    def compute_iou(self, box1, box2):
        """Compute Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_detections(self, image, detections, max_display=10):
        """Draw bounding boxes"""
        print(f"\nDrawing {min(len(detections), max_display)} detections")
        
        for i, det in enumerate(detections[:max_display]):
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_name = det['class_name']
            
            # Color based on class
            color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{i+1}:{class_name[:5]} {score:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"  {i+1}: {class_name} at ({x1},{y1})-({x2},{y2}) score={score:.2f}")
        
        return image
    
    def run(self, image_path, output_path="result.jpg"):
        """Run inference"""
        input_tensor = self.preprocess(image_path)
        
        print(f"\nRunning inference with confidence threshold: {self.conf_threshold}")
        with VDevice() as device:
            params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(self.hef, params)[0]
            
            input_params = InputVStreamParams.make_from_network_group(network_group)
            output_params = OutputVStreamParams.make_from_network_group(network_group)
            
            with network_group.activate(), \
                 InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                
                results = infer_pipeline.infer({self.input_info.name: input_tensor})
                output = results[self.output_info.name]
        
        detections = self.parse_output(output)
        
        # Group by class
        from collections import Counter
        class_counts = Counter([d['class_name'] for d in detections])
        
        print(f"\nFound {len(detections)} valid objects:")
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
        
        # Draw and save
        if detections:
            result_image = self.draw_detections(self.original_image.copy(), detections)
            cv2.imwrite(output_path, result_image)
            print(f"\nResult saved to: {output_path}")
        else:
            print("\nNo valid boxes to draw")
            cv2.imwrite(output_path, self.original_image)
        
        return detections

def main():
    import os
    
    hef_path = "yolov8n.hef"
    image_path = "images.jpeg"
    
    if not os.path.exists(hef_path):
        print(f"Model not found: {hef_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Try with different thresholds
    for thresh in [0.7, 0.5, 0.3]:
        print(f"\n{'='*60}")
        print(f"Testing with confidence threshold: {thresh}")
        print('='*60)
        
        yolo = YOLOv8nHailo(hef_path, conf_threshold=thresh)
        detections = yolo.run(image_path, f"result_thresh_{thresh}.jpg")

if __name__ == "__main__":
    main()