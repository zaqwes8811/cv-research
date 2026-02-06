# Install pipeline (Ubuntu 24.04.1, Hailo8/Yolo5, NVIDIA GeForce RTX 5060 Ti)

0. Setup

```
nvidia-smi
Fri Feb  6 13:27:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off |   00000000:0A:00.0  On |                  N/A |
|  0%   57C    P1             31W /  180W |     465MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1650      G   /usr/lib/xorg/Xorg                       77MiB |
|    0   N/A  N/A            1856      G   /usr/bin/gnome-shell                     20MiB |
|    0   N/A  N/A            4817      C   python                                  332MiB |
+-----------------------------------------------------------------------------------------+
```

1. Install docker, docker compose plugin and docker postinstall

2. Nvidia

```
sudo ubuntu-drivers autoinstall
```

2. Clone project and prepare test dataset

```
# Test installation
cd cv-research
mkdir data && cd data
mkdir datasets

curl -L -o ./labeled-mask-dataset-yolo-darknet.zip \
  https://www.kaggle.com/api/v1/datasets/download/techzizou/labeled-mask-dataset-yolo-darknet

unzip -qq labeled-mask-dataset-yolo-darknet.zip -d source/

python3 ../tidy_data.py

# ./datasets fill be filled
```

2. Train

```
git clone https://github.com/hailo-ai/hailo_model_zoo
cd hailo_model_zoo
git checkout 64a65cbcbc0a80d7e55aca5035c3b2651351bac5 


# ARG base_image=pytorch/pytorch:2.3.0-cuda12.4-cudnn9-runtime
# Exists:
#  ARG base_image=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

cd training/yolov5
docker build -t yolov5_5080ti:v0 .

cd ../../..

mkdir -p hailo/shared

# Temporarily trick the installer
distribution="ubuntu22.04"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

```
tmux new-session -s train
docker stop custom_training
docker rm custom_training

docker run -it --name custom_training --gpus all --ipc=host -v $PWD:/home/hailo/shared yolov5:v0

docker run -it --name custom_training --gpus all --ipc=host -v $PWD:/home/hailo/shared yolov5_5080ti:v0

# Inside docker

sudo apt update
sudo apt install nano -y

apt update && apt install nano -y

cd /home/hailo/shared 

nano data/dataset.yaml

train: ../datasets/images/train
val: ../datasets/images/val
nc: 2
names:
    0: 'using mask'
    1: 'without mask'

python train.py --img 640 --batch 16 --epochs 10 --data dataset.yaml --weights yolov5s.pt

python train.py --img 640 --batch 8  --epochs 1 --data dataset.yaml --weights yolov5s.pt 

python train.py --img 640 --batch 8  --epochs 1 --data dataset.yaml --weights yolov5s.pt --half False


python train.py --img 640 --batch 8  --epochs 1 --data dataset.yaml --weights yolov5s.pt --include onnx

pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow


python3 models/export.py --weights runs/train/exp1/weights/best.pt --img 640 opset=11
python3 export.py --weights runs/train/exp2/weights/best.pt --img 640  --opset 11 --include onnx


# --accumulate 4 # no optioon

wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -q

Trouble:

root@78a17774622c:/workspace/yolov5# python train.py --img 640 --batch 16 --epochs 10 --data dataset.yaml --weights yolov5s.pt
Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex
/opt/conda/lib/python3.8/site-packages/torch/cuda/__init__.py:104: UserWarning: 
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 compute_37.
If you want to use the NVIDIA GeForce RTX 5060 Ti GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Using CUDA device0 _CudaDeviceProperties(name='NVIDIA GeForce RTX 5060 Ti', total_memory=15841MB)

apt update && apt install nano -y

apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libwebp-dev \
    zlib1g-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev

python3 -m venv .venv

RUN git clone https://github.com/hailo-ai/yolov5.git --branch v2.0.1 && \
    cd yolov5 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -U 'coremltools==6.1' 'onnx==1.13.0' 'scikit-learn==0.19.2'

python -m pip install --upgrade pip setuptools

cd yolov5

Cython==0.29.33
numpy==1.21.2
opencv-python==4.5.5.64
#torch==1.7.1
matplotlib==3.6.3
pillow==8.1.0
tensorboard==2.11.2
PyYAML==5.3.1
#torchvision==0.8.2
scipy==1.9.1
tqdm==4.51.0
# pycocotools>=2.0

pip install -U 'coremltools==6.1' 'onnx==1.13.0'
#'scikit-learn==0.19.2' - can't install


nano requirements.txt

commen tourh and torch vision

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130


wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -q

# Remove torch and torch vision


# If using ultralytics pip package
pip install ultralytics --upgrade

# Or clone their fixed version
cd /workspace
git clone https://github.com/ultralytics/yolov5.git yolov5_fixed
cd yolov5_fixed
pip install -r requirements.txt


pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121


https://download.pytorch.org/whl/cu128


```


#####################################################################


1. Build image

```
cd cv-research/5080Ti/

docker build -t yolov5_5080ti:v0 .

tmux new-session -s train

docker stop custom_training && docker rm custom_training
cd ../data/

docker run -it --name custom_training --gpus all --ipc=host -v $PWD:/home/hailo/shared yolov5_5080ti:v0
```

2. Install dataset

```
cd /workspace/yolov5_fixed
cp -R /home/hailo/shared/datasets/ ..

nano data/dataset.yaml

train: ../datasets/images/train
val: ../datasets/images/val
nc: 2
names:
    0: 'using mask'
    1: 'without mask'

```

3. Train

```
. /workspace/.venv/bin/activate
python train.py --img 640 --batch 16 --epochs 10 --data dataset.yaml --weights yolov5s.pt

```

4. Export to `onnx`

```
python export.py --weights runs/train/exp/weights/best.pt --img 640  --opset 11 --include onnx

python detect.py --data dataset.yaml --weights runs/train/exp/weights/best.onnx
python val.py --data dataset.yaml --weights runs/train/exp/weights/best.onnx
```

6. Other NN

```
yolo train data=./data/dataset.yaml model=yolo26n.pt epochs=10 lr0=0.01
cp -R ../datasets/ data/
yolo train data=./data/dataset.yaml model=yolo26n.pt epochs=10 lr0=0.01

python export.py --weights /workspace/yolov5_fixed/runs/detect/train2/weights/best.pt --img 640  --opset 11 --include onnx
```

7. Export to `hef`

```
# Take vagrant from Yandex.Disk
wget https://gist.githubusercontent.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e/raw/20ac954e202fe6a038c2b4bb476703c02fe0df87/ya.py
python3 ya.py https://disk.yandex.ru/d/Rb46K7Oi39jYzw .
```