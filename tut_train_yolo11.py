#https://docs.ultralytics.com/modes/train/

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
# TODO() How to give own dataset
#   https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
results = model.train(data="coco8-grayscale.yaml", epochs=100, imgsz=640)

model.export(format="onnx")