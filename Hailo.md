# Tasks

https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#example-applications

# Haila
- что может она?

https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/

Automatic Optical Inspection
https://hailo.ai/applications/industrial-automation/automatic-optical-inspection/#Anomaly_Detection

# Obj detection alogs

https://hailo.ai/blog/ai-object-detection-on-the-edge-making-the-right-choice/


# [!] HAILO8 

# Impl

https://github.com/hailo-ai

parallel execution
https://community.hailo.ai/t/question-about-running-inference-on-multiple-hailo-models/17777/2

driver
https://github.com/hailo-ai/hailort-drivers

HailoRT
https://hailo.ai/developer-zone/documentation/hailort-v4-22-0/

For newer has no permission

Has yocto layer
https://github.com/hailo-ai/meta-hailo

лучше на базе екты, посмотреть можно ли собрать и запустить? или может убунту и накатить?

[!] https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/hailo-8/python/instance_segmentation

оценщик?
https://hailo.ai/products/hailo-software/model-explorer-vision/

# Hailo emu
https://community.hailo.ai/t/using-hailort-emulator-in-c-code/12855

https://hub.degirum.com/runtime/hailo

https://community.hailo.ai/t/hailo-emulation-failing/14357

no hope for emu

Steps

- https://iambillmeyer.com/posts/2025-01-06-getting-started-with-hailo-ai/

10 by default, need hailo 8

d1af769eb1d8074c5a0151a37b22b46bd483e5a7 - v4.20.0

git checkout tags/v4.20.0

Unknown argument install

cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release && sudo cmake --build build --config release --target install

linux_x86_64


hailortcli scan

sudo hailortcli fw-control identify


./hailort/scripts/download_hefs.sh
~/Work/iglu/hailort/build/hailort/libhailort/examples/cpp

../../hailort/build/hailort/libhailort/examples/cpp/async_infer_advanced_example/cpp_async_infer_advanced_example


# Launch on host

https://github.com/hailo-ai/hailo_model_zoo


git clone https://github.com/hailo-ai/hailo_model_zoo.git

https://community.hailo.ai/t/how-to-install-dataflow-compiler/1276/4
https://community.hailo.ai/t/dataflow-compiler-dfc-availability/1476
https://community.hailo.ai/t/dataflow-compiler-dfc-availability/1476

need account

git checkout tags/v2.17

Component        Requirement                      Found       
==========       ==========                       ==========  ==========
OS               Ubuntu                           Ubuntu      Required
Release          22.04                            24.04       Required
RAM(GB)          16                               19          Required
RAM(GB)          32                               19          Recommended
CPU-Arch         x86_64                           x86_64      Required
CPU-flag         avx                              V           Required
Apt-Package      python3-tk                       X           Required
Apt-Package      graphviz                         X           Required
Apt-Package      libgraphviz-dev                  X           Required
Apt-Package      python3.12-dev                   V           Required
Var:CC           unset                            unset       Required
Var:CXX          unset                            unset       Required
Var:LD           unset                            unset       Required
Var:AS           unset                            unset       Required
Var:AR           unset                            unset       Required
Var:LN           unset                            unset       Required
Var:DUMP         unset                            unset       Required
Var:CPY          unset                            unset       Required
SecureBoot-mode  disabled                         disabled    Required
Apt-Package      linux-headers-6.14.0-36-generic  V           Required
Apt-Package      build-essential                  V           Required
Python           pip                              V           Required
Python           virtualenv                       X           Required

no need for sudo

  182  sudo ./hailo8_ai_sw_suite_2025-10.run 
  183  sudo apt install python3-tk graphviz libgraphviz-dev 
  184  sudo ./hailo8_ai_sw_suite_2025-10.run 
  185  sudo apt install virtualenv


GUIDE:


https://community.hailo.ai/t/hailo-sw-suite-no-gpu-connected/6274/2
  sudo apt install nvidia-cuda-toolkit

Server

sudo apt install python3-tk graphviz libgraphviz-dev python3-pip virtualenv
sudo apt  install cmake
sudo apt install libjpeg-dev zlib1g-dev

on hailo 

Installing collected packages: pyparsing, pydot
  Attempting uninstall: pyparsing
    Found existing installation: pyparsing 3.2.5
    Uninstalling pyparsing-3.2.5:
      Successfully uninstalled pyparsing-3.2.5
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
matplotlib 3.10.7 requires pyparsing>=3, but you have pyparsing 2.4.7 which is incompatible.


virtualenv --python=/usr/bin/python3 venv
Use installed by suite

cd hailo_model_zoo; pip install -e .

fix pyparsing version trouble

Usige:
https://mmmsk.ai.kr/Projects/Embedded-AI/files/hailo_dataflow_compiler_v3.27.0_user_guide.pdf


Validation failed, can't compile cprt file
https://github.com/DeGirum/hailo_examples/blob/main/hailo_model_zoo.md

A .ckpt file is a checkpoint file that stores the state of a machine learning model, most commonly from PyTorch.

# Pure hailo example

Need VPN

https://community.hailo.ai/t/training-a-custom-model-on-rpi-5-8-gb-with-hailo-8-26-tops/7989
https://pub.towardsai.net/custom-dataset-with-hailo-ai-hat-yolo-raspberry-pi-5-and-docker-0d88ef5eb70f?gi=15f20169d3f6

1. Prepare dataset

```

mkdir data && cd data
mkdir datasets

curl -L -o ./labeled-mask-dataset-yolo-darknet.zip \
  https://www.kaggle.com/api/v1/datasets/download/techzizou/labeled-mask-dataset-yolo-darknet

unzip -qq labeled-mask-dataset-yolo-darknet.zip -d source/

python3 ../tidy_data.py
```

2. Train

```
"The training data is small. Training a model from scratch only using this data will produce very poor models, a scenario known as overfitting."  => Transfer learning
```

```
# Now from data folder
git clone https://github.com/hailo-ai/hailo_model_zoo

cd hailo_model_zoo
git checkout 64a65cbcbc0a80d7e55aca5035c3b2651351bac5 

cd training/yolov5
docker build -t yolov5:v0 .

mkdir -p hailo/shared

docker run -it --name custom_training --gpus all --ipc=host -v $PWD:/home/hailo/shared yolov5:v0

# ERROR: docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

Inside docker

sudo apt update
sudo apt install nano -y


python train.py --img 640 --batch 8 --epochs 100 --data dataset.yaml --weights yolov5s.pt
python train.py --img 640 --batch 8 --epochs 10 --data dataset.yaml --weights yolov5s.pt

# Monitor
sudo apt install lm-sensors
sensors  # cli

sudo apt install psensor

cp -R  runs/ /home/hailo/shared/

# Export model

# exp1 for my pc
python3 models/export.py --weights runs/exp1/weights/best.pt --img-size 640 --opset 11  # Failed no opset key

python3 models/export.py --weights runs/exp1/weights/best.pt --img-size 640

or

python3 models/export.py --weights runs/exp1/weights/best.pt --img 640

cp runs/exp0/weights/best.onnx /home/hailo/shared/
```

2. Compile

```

! Use run-file or docker-image from Hailo

# ! Need Hailo account to download some stuff

docker build -t hailo_compiler:v0 .

docker run -it --name compile_onnx_file --gpus all --ipc=host -v $PWD:/home/hailo/shared hailo_compiler:v0


git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
git checkout 64a65cbcbc0a80d7e55aca5035c3b2651351bac5   # < Need it
pip install -e .

```

3. Install with run

```
sudo apt install python3-tk graphviz libgraphviz-dev
./hailo8_ai_sw_suite_2025-10.run 
```

4. Docker

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L \
https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
| sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

hailomz compile --ckpt ./best.onnx --yaml hailo_model_zoo/cfg/networks/yolov5s.yaml  --classes 2 --hw-arch hailo8

cd /local/
hailomz compile --ckpt ./best.onnx --yaml hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov5s.yaml  --calib-path ./datasets/images/train/ --classes 2 --hw-arch hailo8