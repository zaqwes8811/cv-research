# cv-research

Задача:

Пока главное оценить самые тяжелые операции

https://habr.com/ru/companies/data_light/articles/855336/

- Выделить - семантическая сегментация (возможно нужны больше датасеты? попиксельная классификация)
- Понять что это - классицифкация
- совместить с эталоном? измерить как?

Трудности оценки:
- Искажения перспективы - сцена в 3д, и возможно нужна калибровка
- совмещение образца с выделенным объектом
- могут ли в зоне видимости быть еще объекты?
- [!] или просто классификация - плохая фераз-хорошая фреза, и сравнивать не нужно, и не то чтобы измерения были
- калибровка, а если объеманя история, или не на столе

https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#example-applications

Instance Segmentation - и класс и занимаемое поле - выглядит очень не точно

# Архитектуры

Если сегментация:
- FCN
- U-Nets
- DeepLab

Но если только детектирование и классификация:
- YoLo
- Mask R-CNN

https://github.com/ultralytics/ultralytics
https://docs.ultralytics.com/ru/tasks/classify/


https://habr.com/ru/articles/721414/

FCN.ResNet101

https://www.kaggle.com/code/growinfame/semantic-segmentation-tutorial-pytorch-lightning

Обзор
https://elar.urfu.ru/bitstream/10995/140348/1/m_th_v.a.melnikov_2024.pdf

# PyTorch tutorial

https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

virtualenv --python=/usr/bin/python3.10 venv

virtualenv --python=/usr/bin/python3 venv

pip install lightning  # install lot of stuff
pip install segmentation-models-pytorch
pip install albumentations
pip install matplotlib
pip install tensorflow
pip install ultralytics
pip install onnx onnxslim onnxruntime-gpu


HW: Kaggle выделят бесплатные gpu ресурсы?

TF_ENABLE_ONEDNN_OPTS=0 python sem_segm_kaggle_tut.py

# Nvidia driver
lspci | grep VGA

https://stackoverflow.com/questions/54264338/why-does-pytorch-not-find-my-nvdia-drivers-for-cuda-support

sudo apt-get install nvidia-cuda-toolkit

nvidia-detector

sudo apt install nvidia-utils-580
sudo apt install nvidia-driver-580

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

# VSCode

ssh ...@.. to Add and than password

