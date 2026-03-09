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
        
        # Convert BGR to RGB and resize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # Convert HWC to CHW and add batch dimension
        input_tensor = np.transpose(resized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
        
        return input_tensor
    
    def parse_output(self, output):
        """Parse the model output - with correct coordinate indices"""
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
                            # Try different coordinate orders
                            # Option 1: [x1, y1, x2, y2, score]
                            # Option 2: [x_center, y_center, width, height, score]
                            # Option 3: [y1, x1, y2, x2, score]
                            
                            # Let's print the first detection to debug
                            if det_idx == 0 and class_id == 0:  # First person detection
                                print(f"  Raw detection values: {det}")
                            
                            # Try different interpretations
                            score = float(det[4])  # Score is usually the 5th value
                            
                            if score > self.conf_threshold:
                                # Try to interpret coordinates - adjust these indices based on your output
                                # Currently getting [0,0,0,0] means we need the right indices
                                
                                # Let's try assuming normalized coordinates [x1, y1, x2, y2]
                                x1 = float(det[0])
                                y1 = float(det[1])
                                x2 = float(det[2])
                                y2 = float(det[3])
                                
                                # If all zeros, maybe they're in different positions
                                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                                    # Try [x_center, y_center, width, height]
                                    x_center = float(det[0])
                                    y_center = float(det[1])
                                    width = float(det[2])
                                    height = float(det[3])
                                    
                                    if x_center > 0 or y_center > 0 or width > 0 or height > 0:
                                        # Convert center format to corner format
                                        x1 = x_center - width/2
                                        y1 = y_center - height/2
                                        x2 = x_center + width/2
                                        y2 = y_center + height/2
                                
                                # Scale coordinates to original image
                                x1 = int(x1 * self.original_width / self.input_width)
                                y1 = int(y1 * self.original_height / self.input_height)
                                x2 = int(x2 * self.original_width / self.input_width)
                                y2 = int(y2 * self.original_height / self.input_height)
                                
                                # Ensure coordinates are valid
                                x1 = max(0, min(x1, self.original_width))
                                y1 = max(0, min(y1, self.original_height))
                                x2 = max(0, min(x2, self.original_width))
                                y2 = max(0, min(y2, self.original_height))
                                
                                # Only add if box has positive area
                                if x2 > x1 and y2 > y1:
                                    detections.append({
                                        'bbox': [x1, y1, x2, y2],
                                        'score': score,
                                        'class_id': class_id,
                                        'class_name': self.class_names[class_id]
                                    })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        for det in detections:
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
                output = results[self.output_info.name]
        
        # Parse detections
        detections = self.parse_output(output)
        print(f"\nFound {len(detections)} valid objects:")
        
        # Group by class
        from collections import Counter
        class_counts = Counter([d['class_name'] for d in detections])
        for class_name, count in class_counts.most_common():
            print(f"  {class_name}: {count}")
        
        # Show individual detections
        if detections:
            print("\nDetections with valid boxes:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']}: {det['score']:.2f} at {det['bbox']}")
        
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
    
    # Configuration
    hef_path = "yolov8n.hef"
    image_path = "output_640x640.jpg"
    
    if not os.path.exists(hef_path):
        print(f"Model not found: {hef_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Test with different thresholds
    thresholds = [0.3, 0.2, 0.1]
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing with confidence threshold: {thresh}")
        print('='*60)
        
        yolo = YOLOv8nHailo(hef_path, conf_threshold=thresh)
        detections = yolo.run(image_path, f"result_thresh_{thresh}.jpg")
        
        if detections:
            print(f"\n✓ Found {len(detections)} valid objects with threshold {thresh}")
            break
        else:
            print(f"✗ No valid objects found with threshold {thresh}")

if __name__ == "__main__":
    main()