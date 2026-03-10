#!/usr/bin/env python3
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams

class YOLOv8nHailo:
    def __init__(self, hef_path, conf_threshold=0.3):
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
        
        self.input_height, self.input_width = self.input_info.shape[:2]
        print(f"  Model input size: {self.input_width} x {self.input_height}")
    
    def letterbox_image(self, image, target_size):
        """Resize image with letterboxing to maintain aspect ratio"""
        height, width = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale and padding
        scale = min(target_w / width, target_h / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size (using padding color 114, common for YOLO)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Store letterbox info for coordinate transformation
        self.letterbox_info = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'original_width': width,
            'original_height': height
        }
        
        return canvas
    
    def preprocess(self, image_path):
        """Preprocess image with letterboxing"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = img.copy()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply letterboxing to maintain aspect ratio
        letterboxed = self.letterbox_image(img_rgb, (self.input_width, self.input_height))
        
        # Convert HWC to CHW and add batch dimension
        input_tensor = np.transpose(letterboxed, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
        
        print(f"Original image: {self.original_image.shape[1]} x {self.original_image.shape[0]}")
        print(f"Letterbox info: scale={self.letterbox_info['scale']:.3f}, "
              f"pad=({self.letterbox_info['pad_w']}, {self.letterbox_info['pad_h']})")
        
        return input_tensor
    
    def transform_coordinates(self, x_norm, y_norm):
        """Transform normalized coordinates back to original image space"""
        # First, scale to letterboxed image coordinates
        x_box = x_norm * self.input_width
        y_box = y_norm * self.input_height
        
        # Remove padding
        x_box -= self.letterbox_info['pad_w']
        y_box -= self.letterbox_info['pad_h']
        
        # Scale back to original image size
        x_orig = x_box / self.letterbox_info['scale']
        y_orig = y_box / self.letterbox_info['scale']
        
        # Clip to image boundaries
        x_orig = max(0, min(x_orig, self.letterbox_info['original_width']))
        y_orig = max(0, min(y_orig, self.letterbox_info['original_height']))
        
        return int(x_orig), int(y_orig)
    
    def parse_output(self, output):
        """Parse the model output with proper coordinate transformation"""
        detections = []
        
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]
            
            for class_id, class_output in enumerate(batch_output):
                if class_id >= len(self.class_names):
                    continue
                
                if isinstance(class_output, np.ndarray) and class_output.size > 0:
                    for det_idx, det in enumerate(class_output):
                        if len(det) >= 5:
                            # Get normalized coordinates and score
                            x1_norm = float(det[0])
                            y1_norm = float(det[1])
                            x2_norm = float(det[2])
                            y2_norm = float(det[3])
                            score = float(det[4])
                            
                            if score > self.conf_threshold:
                                # Transform coordinates to original image space
                                x1, y1 = self.transform_coordinates(x1_norm, y1_norm)
                                x2, y2 = self.transform_coordinates(x2_norm, y2_norm)
                                
                                # Ensure x1 < x2 and y1 < y2
                                if x1 > x2:
                                    x1, x2 = x2, x1
                                if y1 > y2:
                                    y1, y2 = y2, y1
                                
                                # Calculate area to filter tiny boxes
                                area = (x2 - x1) * (y2 - y1)
                                
                                # Only add if box has reasonable size
                                if area > 500:  # Minimum area threshold
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'score': score,
                                        'class_id': class_id,
                                        'class_name': self.class_names[class_id],
                                        'area': area
                                    })
        
        # Sort by score and area (higher score, larger area first)
        detections.sort(key=lambda x: (x['score'], x['area']), reverse=True)
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        print(f"\nDrawing {len(detections)} detections")
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_name = det['class_name']
            
            # Generate color based on class
            color = (0, 255, 0)  # Green for all
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{i+1}:{class_name[:5]} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, (x1, y1-15), (x1+w, y1), color, -1)
            cv2.putText(image, label, (x1+2, y1-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return image
    
    def run(self, image_path, output_path="result.jpg"):
        """Run inference on image"""
        # Preprocess with letterboxing
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
                output = results[self.output_info.name]
        
        # Parse detections
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
            
            # Also save an image with just the top 5 detections for clarity
            top5_image = self.original_image.copy()
            for det in detections[:5]:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(top5_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite("top5_detections.jpg", top5_image)
        else:
            print("\nNo valid boxes to draw")
        
        return detections

def main():
    import os
    
    hef_path = "yolov8n.hef"
    image_path = "images.jpg"
    
    if not os.path.exists(hef_path):
        print(f"Model not found: {hef_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test with different thresholds
    thresholds = [0.5, 0.3]
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with confidence threshold: {thresh}")
        print('='*60)
        
        yolo = YOLOv8nHailo(hef_path, conf_threshold=thresh)
        detections = yolo.run(image_path, f"result_thresh_{thresh}.jpg")
        
        if detections:
            print(f"\n✓ Found {len(detections)} objects with threshold {thresh}")

if __name__ == "__main__":
    main()