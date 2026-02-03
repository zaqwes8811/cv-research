
# Yollo
https://www.kaggle.com/code/denizcanelci/yoloworld-tutorial-for-beginners

Trouble
Python 12 on work station

virtualenv --python=/usr/bin/python3.12 venv_3.12
sudo apt install python3-distutils # Doesn't work
python3 -m pip install setuptools

https://github.com/open-mmlab/mim/issues/242

https://stackoverflow.com/questions/77364550/attributeerror-module-pkgutil-has-no-attribute-impimporter-did-you-mean

`sudo apt-get install python3-venv`

-> 3.10

virtualenv --python=/usr/bin/python3.10 venv_3.10
pip install -U setuptools # no need second time
pip install --upgrade inference  # failed


# Pure train
https://docs.ultralytics.com/modes/train/

https://docs.ultralytics.com/datasets/detect/coco8/

!! https://community.hailo.ai/t/retrain-yolov5-on-a-custom-dataset/1552


# General tips

sudo apt install python3.10-dev

python -m pip install "pycocotools @ git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"

export SETUPTOOLS_USE_DISTUTILS=stdlib

https://github.com/open-mmlab/mmdetection/issues/1677

pip uninstall setuptools

curl https://files.pythonhosted.org/packages/69/77/aee1ecacea4d0db740046ce1785e81d16c4b1755af50eceac4ca1a1f8bfd/setuptools-60.5.0.tar.gz > setuptools-60.5.0.tar.gz

tar -xzvf setuptools-60.5.0.tar.gz

cd setuptools-60.5.0

python3 setup.py install


cocodataset

pip install fiftyone

import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
“coco-2017”,
splits=[“train”, “validation”, “test”],
label_type=“detections”,
classes=[“person”, “car”, “bicycle”],
)
