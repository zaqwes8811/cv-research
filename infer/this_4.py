#!/usr/bin/env python3
import cv2
import numpy as np
import math
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
        
        self.input_height, self.input_width = self.input_info.shape[:2]
        print(f"Model loaded: {self.input_width} x {self.input_height}")
        print(f"Output shape info: {self.output_info.shape}")
    
    def preprocess(self, image_path):
        """Preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = img.copy()
        self.orig_h, self.orig_w = img.shape[:2]
        
        print(f"Original image: {self.orig_w} x {self.orig_h}")
        
        # Calculate scale factor
        self.scale_factor = self.input_width / self.orig_w
        
        # Convert BGR to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # Convert to CHW and add batch dimension
        input_tensor = np.transpose(resized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
        
        return input_tensor
    
    def parse_output(self, output):
        """Parse the model output - standard YOLOv8 format [x1, y1, x2, y2, confidence]"""
        detections = []
        
        if isinstance(output, list) and len(output) > 0:
            batch_output = output[0]  # List of 80 class outputs
            
            for class_id, class_output in enumerate(batch_output):
                if class_id >= len(self.class_names):
                    continue
                
                if isinstance(class_output, np.ndarray) and class_output.size > 0:
                    # class_output is array of shape [num_detections, 5]
                    for det in class_output:
                        print(det)
                        if len(det) >= 5:
                            # Extract values - all normalized 0-1
                            x1_norm, y1_norm, x2_norm, y2_norm, confidence = det[:5]
                            
                            if confidence > self.conf_threshold:
                                # Convert normalized coordinates to original image pixels
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
        
        return detections
    def draw_detections(self, image, detections, max_display=20):
        """Draw bounding boxes on image - with rotation visualization"""
        print(f"\nDrawing {len(detections)} detections")
        
        for i, det in enumerate(detections[:max_display]):
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_name = det['class_name']
            is_rotated = det.get('rotated', False)
            angle = det.get('angle', 0)
            
            # Color based on class
            if class_name == 'person':
                color = (0, 255, 0)  # Green
            elif class_name == 'snowboard':
                color = (255, 0, 0)  # Blue
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw the axis-aligned bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # If rotated, draw a small indicator
            if is_rotated:
                # Draw a small circle and line to show rotation
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
                # Draw a line showing rotation direction
                line_end_x = center_x + int(30 * math.cos(math.radians(angle)))
                line_end_y = center_y + int(30 * math.sin(math.radians(angle)))
                cv2.line(image, (center_x, center_y), (line_end_x, line_end_y), (0, 0, 255), 2)
            
            # Draw label
            rot_marker = " R" if is_rotated else ""
            label = f"{i+1}:{class_name[:4]}{rot_marker} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, (x1, y1-15), (x1+w, y1), color, -1)
            cv2.putText(image, label, (x1+2, y1-3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            print(f"  {i+1}: {class_name} at ({x1},{y1})-({x2},{y2})" + (" (rotated)" if is_rotated else ""))
        
        return image
    
    def run(self, image_path, output_path="result.jpg"):
        """Run inference on image"""
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
            rotated_count = sum(1 for d in detections if d['class_name'] == class_name and d.get('rotated', False))
            print(f"  {class_name}: {count} (rotated: {rotated_count})")
        
        # Draw and save
        if detections:
            result_image = self.draw_detections(self.original_image.copy(), detections)
            cv2.imwrite(output_path, result_image)
            print(f"\nResult saved to: {output_path}")
        else:
            print("\nNo valid boxes to draw")
        
        return detections

def main():
    import os
    
    hef_path = "yolov8n.hef"
    image_path = "wine_glass_ubc.jpg"
    
    if not os.path.exists(hef_path):
        print(f"Model not found: {hef_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Try multiple thresholds to see effect
    for thresh in [0.4]:
        print(f"\n{'='*60}")
        print(f"Testing with confidence threshold: {thresh}")
        print('='*60)
        
        yolo = YOLOv8nHailo(hef_path, conf_threshold=thresh)
        detections = yolo.run(image_path, f"result_thresh_{thresh}.jpg")
        
        if detections:
            print(f"\n✓ Found {len(detections)} objects with threshold {thresh}")

if __name__ == "__main__":
    main()