import warnings
import numpy as np
import pandas as pd
import random
from osgeo import gdal
from PIL import Image
import config
config=config.config

def resample_tif(img_re):
    """
    :param img: original factors data
    :return: resampled factors data
    """
    warnings.filterwarnings("ignore")
    img_re = np.array(Image.fromarray(img_re).resize((config["height"], config["width"])))
    return img_re

def read_dada_from_tif(tif_path):
    """
    读取影响因子数据并转换为nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    if(w != config["width"] and h != config["height"]):
        img = resample_tif(img)
    return img

def get_feature_data():
    """"
    读取特征并进行min-max归一化
    """
    tif_paths = config["data_path"]
    data = np.zeros((config["feature"], config["width"], config["height"])).astype(np.float32)
    for i, tif_path in enumerate(tif_paths):
        img = read_dada_from_tif(tif_path)
        data[i,:,:] = (img-0)/config["data_max"][i]
        # data[i,:,:] = img[i]
        # print(np.unique(data))
    return  data

def save_to_excel(numpy_data, file):
    """
    保存中间过程
    """
    data = np.squeeze(np.array(numpy_data))
    data = pd.DataFrame(data)

    writer = pd.ExcelWriter(file) 
    data.to_excel(writer, 'page_1', float_format='%.8f')  
    writer.save()
    writer.close()
    return 0

class creat_dataset():
    """          
    以滑坡点为中心扩展滑坡为 N×N，N最好为基数
    """
    def __init__(self,tensor_data, n):
        self.data = tensor_data
        self.n = int(n)
        self.p = int((n-1)/2)
        self.F = tensor_data.shape[0]
        self.w = tensor_data.shape[1]
        self.h = tensor_data.shape[2]

    def creat_new_tensor(self):
        new_tensor = np.zeros((self.F, self.w + self.n -1, self.h +self.n -1))
        new_tensor[:, self.p:self.w+self.p, self.p:self.h+self.p] = self.data
        return new_tensor
    
    def pixel_to_image(self, data):
        images = []
        for i in range(config["width"]):
            for j in range(config["height"]):
                images.append(data[:, i:i+self.n, j:j+self.n].astype(np.float32))
        return np.stack(images)
    
    def get_images_labels(self, data, labels, mode='train'):
        train_images, train_labels = [], []
        valid_images, valid_labels = [], []
        count_0, count_1, count_2, count_3,count_4 = 0, 0, 0, 0,0
        for i in range(config["width"]):
            for j in range(config["height"]):
                if(labels[i,j]==0 or labels[i,j]==2):
                    train_images.append(data[:, i:i+self.n, j:j+self.n].astype(np.float32))
                    if (labels[i,j]==0):
                        count_0 += 1
                        train_labels.append(0)
                    if (labels[i,j]==2):
                        count_2 += 1
                        train_labels.append(1)   
                if(labels[i,j]==1 or labels[i,j]==3):
                    valid_images.append(data[:, i:i+self.n, j:j+self.n].astype(np.float32))
                    if (labels[i,j]==1):
                        count_1 += 1
                        valid_labels.append(0)
                    if (labels[i,j]==3):
                        count_3 += 1
                        valid_labels.append(1)
        print("label 为 0，1，2，3, 4的像素点个数分别为{},{},{},{},{}".format(count_0, count_1, count_2, count_3, count_4))
        if(mode=="train"):
            print('训练集： '+str(len(train_images)), str(len(train_labels)))
            return train_images, train_labels
        else:
            print('测试集： '+str(len(valid_images)), str(len(valid_labels)))
            return valid_images, valid_labels

def read_label():
    """
    :读取栅格数据
    :返回标签列表
    """
    label = read_dada_from_tif(config["label_path"])
    labels = []
    count_0, count_1, count_2, count_3,count_4 = 0, 0, 0, 0,0
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):

            if(label[i,j] == 0):
                labels.append(0)
                count_0 += 1

            elif(label[i,j] == 1):
                labels.append(1)
                count_1 += 1

            elif(label[i,j] == 2):
                labels.append(2)
                count_2 += 1

            elif(label[i,j] == 3):
                labels.append(3)
                count_3 += 1

            elif(label[i,j] == 4):
                labels.append(4)
                count_4 += 1

            else:
                labels.append(-1)
    # save_to_excel(label, "./l.xlsx")
    print("label 为 0，1，2，3, 4的像素点个数分别为{},{},{},{},{}".format(count_0, count_1, count_2, count_3, count_4))
    return labels

def split_0_1_2_3(images, labels, mode="train"):
    """
    :0 and 1 represent the non-landslide and landslide of the training dataset, respectively.
    :2 and 3 represent the non-landslide and landslide of the validation dataset, respectively.
    :images: size [N, 12, w, h],
    :labels: size [N],  N = w*h
    """
    train_images, train_labels = [], []
    valid_images, valid_labels = [], []
    for i in range(len(labels)):

        if(labels[i]==0 or labels[i]==2):
            train_images.append(images[i][:,:,:])
            if (labels[i]==0):
                 train_labels.append(labels[i])
            if (labels[i]==2):
                train_labels.append(1)
        elif(labels[i]==1 or labels[i]==3):
            valid_images.append(images[i][:,:,:])
            if (labels[i]==1):
                valid_labels.append(0)
            if (labels[i]==3):
                valid_labels.append(1)

    if(mode=="train"):
        print('训练集： '+str(len(train_images)), str(len(train_labels)))
        return train_images, train_labels
    else:
        print('测试集： '+str(len(valid_images)), str(len(valid_labels)))
        return valid_images, valid_labels

def shuffle_image_label_0(images, labels):
    """
    Randomly disrupt two list with the same shuffle
    """
    randnum = random.randint(0, len(images))
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels


def train_data():
    tensor_data = get_feature_data()  #读取训练数据 9*3368*2626
    creat = creat_dataset(tensor_data, config["size"])  
    data = creat.creat_new_tensor()   #生成

    labels = read_dada_from_tif(config["label_path"])
    train_images, train_labels = creat.get_images_labels(data, labels, mode='train')

    train_images, train_labels = shuffle_image_label_0(train_images, train_labels)
    return np.array(train_images).reshape((-1,config["feature"],config["size"],config["size"])), np.array(train_labels).reshape((-1,1))


def test_data():
    tensor_data = get_feature_data()
    creat = creat_dataset(tensor_data, config["size"])
    data = creat.creat_new_tensor()

    labels = read_dada_from_tif(config["label_path"])
    images, labels = creat.get_images_labels(data, labels, mode='valid')

    # print("训练集的 label 为0,1的个数比例{}：{} = {}".format(len(images_0), len(images_1), len(images_0)/len(images_1)))

    return np.array(images).reshape((-1,config["feature"],config["size"],config["size"])), np.array(labels).reshape((-1,1))

def pred_data():
    """
     data sets for 整个研究区域
    """
    tensor_data = get_feature_data()
    print(tensor_data.shape)
    creat = creat_dataset(tensor_data, config["size"])
    data = creat.creat_new_tensor()
    data = creat.pixel_to_image(data)
    print('ok')
    return data

def save_to_tif(pred_result, save_path):
    """
    :保存LSM
    """
    img = pred_result.reshape((config["width"], config["height"]))
    im_geotrans, im_prof = [], []
    for tif_path in config["data_path"]:#取仿射矩阵、投影坐标
        tif = gdal.Open(tif_path)
        im_geotrans.append(tif.GetGeoTransform())
        im_prof.append(tif.GetProjection())

    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    #判读数组维数
    if len(img.shape) == 3:
        im_bands, im_height, im_width = img.shape
    else:
        im_bands, (im_height, im_width) = 1,img.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(save_path, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans[-1])              #写入仿射变换参数
    dataset.SetProjection(im_prof[-1])                    #写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(img[i])
    del dataset
    print('ok')


    




