# Tut: https://www.kaggle.com/code/growinfame/semantic-segmentation-tutorial-pytorch-lightning/notebook

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as pl
from glob import glob
import segmentation_models_pytorch as smp
from lightning.pytorch.callbacks import ModelCheckpoint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn.functional as F