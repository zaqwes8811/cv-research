# Dockerfile for RTX 5080 Ti (sm_120) + YOLOv5
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL description="YOLOv5 training on RTX 5080 Ti with CUDA 12.8"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN cd /workspace && python3 -m venv .venv

RUN . .venv/bin/activate && git clone https://github.com/ultralytics/yolov5.git yolov5_fixed \
    && cd yolov5_fixed \
    && pip install -r requirements.txt \
    && pip install ultralytics --upgrade \
    && wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -q

# Expose ports
EXPOSE 8888 6006

CMD ["bash"]


