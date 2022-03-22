import gdal
import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import draw
from sklearn.utils import shuffle
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import check_non_negative
import torch
import sys
from torch import optim
from torch.functional import Tensor
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.pooling import MaxPool2d
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from torchvision.transforms.functional import resize
#from logger import Logger
from sklearn.metrics import roc_curve, auc, confusion_matrix,f1_score
import pylab as plt
import numpy as np
from scipy import interp
from itertools import cycle
import os
from tqdm import tqdm, trange  #进度条

def read_img(target_img):
    dataset = gdal.Open(target_img)
    width = dataset.RasterXSize #col
    height = dataset.RasterYSize #row
    bands = dataset.RasterCount
    im_data = dataset.ReadAsArray()
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    im_data[np.isinf(im_data)] = 0
    im_data[np.isneginf(im_data)] = 0
    im_data[np.isnan(im_data)] = 0
    del dataset
    return width, height, bands, im_data, geotrans, proj

target_img =  'D:/CNN/study/11波段.tif'
width_old, height_old, bands, data_old, geotrans_old, proj = read_img(target_img)

def extend_img(data_old):
    old_data = data_old.swapaxes(0,1)
    old_data = old_data.swapaxes(1,2)
    new_data = np.pad(old_data, ((17,17),(17,17),(0,0)),'constant', constant_values=((0,0),(0,0),(0,0)))
    # new_data = new_data.swapaxes(1,2)
    # new_data = new_data.swapaxes(0,1)
    return new_data

class SELayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)

class cnn_mul(nn.Module):
    def __init__(self):
        super(cnn_mul, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=11,
                out_channels=64,
                kernel_size=3,
                stride=1,

                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),

            nn.MaxPool2d(kernel_size=2),
        )
        self.se1 = SELayer(channel=64, reduction=16)

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),

            nn.MaxPool2d(kernel_size=2),
        )
        self.se2 = SELayer(channel=128, reduction=16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=196,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=196),
            nn.ReLU(),

        )
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=196 * 4 * 4, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=2)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)

        # x=self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        output = self.fc2(x)

        return output, x

#重写dataset
class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                                 std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))  # 归一化
            ])

    def __getitem__(self, item):
        img, cols_rows = self.data_list[item]
        return img, cols_rows

    def __len__(self):
        return len(self.data_list)

def k():
    k = np.ones((17, 17, 11))
    # k = k.swapaxes(1,2)
    # k = k.swapaxes(0,1)
    return k

#重写dataset

ex = extend_img(data_old)
k = k()





def temp_img(the_img,the_idx):
    img = the_img
    idx = the_idx
    img = img.swapaxes(1,2)
    img = img.swapaxes(0,1)
    return img,idx

array_list = []
for col in range(width_old):
    if col%100 == 0:
        print(col)
    for row in range(height_old):
        temp = ex[row:row+17,col:col+17,:]
        temp_array = temp * k
        col_row = str(col)+"_"+str(row)
        temp_ = temp_img(temp_array,col_row)
        array_list.append(temp_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('cnn1.pkl')
model.to(device)

image_pro = np.zeros([height_old, width_old])
image_pre = np.zeros([height_old, width_old])

all_data = MyDataset(array_list)
#将预测数据送入模型
pre_data = DataLoader(all_data, 32,shuffle=False)
for pre_img, pre_col_row in pre_data:
    model.eval()  # 改变模型状态
    #print(pre_col_row)
    the_image = pre_img.to(device)
    outputs = model(the_image)[0]
    class_out = torch.max(outputs.data, 1)[1].data.cpu().numpy()
    class_pro = outputs.detach().cpu().numpy()
    class_pro=softmax(class_pro)

    for c_r in range(len(pre_col_row)):
        array_col,array_row = int(pre_col_row[c_r].split("_")[0]) ,int(pre_col_row[c_r].split("_")[1])
        image_pre[array_row,array_col] = class_out[c_r]
        image_pro[array_row,array_col] = class_pro[c_r][1]

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

path1 = 'D:/CNN/study/pre3.tif'
path2 = 'D:/CNN/study/pro3.tif'
writeTiff(image_pre, geotrans_old, proj, path1)
writeTiff(image_pro, geotrans_old, proj, path2)
