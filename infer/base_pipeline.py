from hailo_platform import VDevice, HEF
import numpy as np
hef = HEF('yolov8n.hef')
with VDevice() as dev:
    net = dev.configure(hef)[0]
    with net.activate():
        print('✓ Inference pipeline ready')