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

# VSCode

ssh ...@.. to Add and than password

