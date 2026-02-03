import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

# HANGs
#model = YOLOWorld(model_id="yolo_world/l")