import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import lightgbm
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from __future__ import  absolute_import
from __future__ import  division
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
from utils.config import opt
import json

AMAP_LABEL_NAMES = (
    'unblocked',
    'slow',
    'blocked',
    )

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)

        return img, label, scale

class AMAPDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_names = AMAP_LABEL_NAMES
        self.ids = os.listdir(self.data_dir + 'amap_traffic_train_0712')

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):

        id_ = self.ids[i]
       
        with open(self.data_dir+"amap_traffic_annotations_train.json","r") as f:
            content=f.read()
        content=json.loads(content)
        cid = content['annomation'][i]

        if id_ == cid['id']:      
            img_file = os.path.join(self.data_dir, 'amap_traffic_train_0712', cid['id'], cid['key_frame'])
            # print(img_file)
            image = read_image(img_file, color=True)
            # img.append(image)
            label = (cid['status'])
    
        return image, label
    __getitem__ = get_example

class AMAPTestDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_names = AMAP_LABEL_NAMES
        self.ids = os.listdir(self.data_dir + 'amap_traffic_test_0712')

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):

        id_ = self.ids[i]
       
        with open(self.data_dir+"amap_traffic_annotations_test.json","r") as f:
            content=f.read()
        content=json.loads(content)
        cid = content['annomation'][i]

        if id_ == cid['id']:      
            img_file = os.path.join(self.data_dir, 'amap_traffic_test_0712', cid['id'], cid['key_frame'])
            # print(img_file)
            image = read_image(img_file, color=True)
            # img.append(image)
            label = (cid['status'])
    
        return image, label
    __getitem__ = get_example

class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = AMAPDataset(opt.amap_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, label = self.db.get_example(idx)
        img, label, scale = self.tsf((ori_img, label))
        return img.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

class TestDataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = AMAPTestDataset(opt.amap_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, label = self.db.get_example(idx)
        img, label, scale = self.tsf((ori_img, label))
        return img.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

    # with open(result_path+"sub_%s.json"%m,"w") as f:
    #     f.write(json.dumps(content))


# json_path = "amap_traffic_annotations_test.json"
# out_path = "amap_traffic_annotations_test_result.json"

# # result 是你的结果, key是id, value是status
# with open(json_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
#     json_dict = json.load(f)
#     data_arr = json_dict["annotations"]  
#     new_data_arr = [] 
#     for data in data_arr:
#         id_ = data["id"]
#         data["status"] = int(result[id_])
#         new_data_arr.append(data)
#     json_dict["annotations"] = new_data_arr
#     json.dump(json_dict, w)
