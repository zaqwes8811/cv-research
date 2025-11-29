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