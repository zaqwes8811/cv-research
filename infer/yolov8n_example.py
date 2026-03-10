#!/usr/bin/env python3
"""
YOLOv8n inference on Hailo-8
Run with: hailo-python yolov8n_example.py
"""
import cv2
import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

class YOLOv8nHailo:
    def __init__(self, hef_path, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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
        
        # YOLOv8 expects 640x640 input
        self.input_height = self.input_info.shape[1]
        self.input_width = self.input_info.shape[2]
        
    def preprocess(self, image_path):
        """Preprocess image for YOLOv8"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        self.original_image = img.copy()
        self.original_height, self.original_width = img.shape[:2]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with letterbox (maintain aspect ratio)
        scale = min(self.input_width / self.original_width, 
                   self.input_height / self.original_height)
        new_width = int(self.original_width * scale)
        new_height = int(self.original_height * scale)
        
        resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Create canvas and paste resized image
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        x_offset = (self.input_width - new_width) // 2
        y_offset = (self.input_height - new_height) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Normalize to [0, 1] and convert to float32
        canvas = canvas.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format for Hailo
        canvas = np.transpose(canvas, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = np.expand_dims(canvas, axis=0)
        
        # Store letterbox info for postprocessing
        self.letterbox_info = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'original_width': self.original_width,
            'original_height': self.original_height
        }
        
        return input_tensor
    
    def postprocess(self, output):
        """Postprocess YOLOv8 output"""
        # Remove batch dimension and get predictions
        predictions = output[0]  # Shape: [84, 8400]
        
        # Transpose to [8400, 84]
        predictions = np.transpose(predictions)
        
        # Split into boxes, scores, classes
        boxes = predictions[:, :4]
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)
        
        # Filter by confidence
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Rescale boxes to original image coordinates
        scale = self.letterbox_info['scale']
        x_offset = self.letterbox_info['x_offset']
        y_offset = self.letterbox_info['y_offset']
        orig_w = self.letterbox_info['original_width']
        orig_h = self.letterbox_info['original_height']
        
        # Remove padding and scale
        x1 = (x1 - x_offset) / scale
        y1 = (y1 - y_offset) / scale
        x2 = (x2 - x_offset) / scale
        y2 = (y2 - y_offset) / scale
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Convert to int
        boxes = np.column_stack([x1, y1, x2, y2]).astype(int)
        
        # Apply NMS (simple version)
        indices = self.nms(boxes, scores)
        
        results = []
        for i in indices:
            results.append({
                'bbox': boxes[i],
                'score': scores[i],
                'class_id': class_ids[i],
                'class_name': self.class_names[class_ids[i]]
            })
        
        return results
    
    def nms(self, boxes, scores):
        """Simple Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def draw_results(self, image, results):
        """Draw bounding boxes on image"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            score = result['score']
            class_name = result['class_name']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image
    
    def run(self, image_path):
        """Run inference on image"""
        # Preprocess image
        input_tensor = self.preprocess(image_path)
        
        # Run inference on Hailo
        print("Running inference on Hailo-8...")
        with VDevice() as device:
            # Configure network group
            params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(self.hef, params)[0]
            
            # Create stream parameters
            input_params = InputVStreamParams.make_from_network_group(network_group)
            output_params = OutputVStreamParams.make_from_network_group(network_group)
            
            # Activate and run inference
            with network_group.activate(), \
                 InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                
                # Run inference
                input_dict = {self.input_info.name: input_tensor}
                results = infer_pipeline.infer(input_dict)
                output = results[self.output_info.name]
        
        print("Inference complete!")
        
        # Postprocess
        detections = self.postprocess(output)
        print(f"Found {len(detections)} objects:")
        for det in detections:
            print(f"  {det['class_name']}: {det['score']:.2f} at {det['bbox']}")
        
        # Draw results
        result_image = self.draw_results(self.original_image.copy(), detections)
        
        return result_image, detections

def main():
    # Configuration
    hef_path = "yolov8n.hef"
    image_path = "test_image.jpg"  # Replace with your image
    output_path = "result.jpg"
    
    # Check if files exist
    import os
    if not os.path.exists(hef_path):
        print(f"ERROR: Model file not found: {hef_path}")
        print("Download with: wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef")
        return
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        print("Please provide an image file")
        return
    
    # Initialize YOLO
    yolo = YOLOv8nHailo(hef_path)
    
    # Run inference
    result_image, detections = yolo.run(image_path)
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()