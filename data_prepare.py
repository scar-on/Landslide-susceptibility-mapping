from osgeo import gdal
import numpy as np
import pandas as pd
import os
import random


def read_data_from_tif(tif_path):
    """
    Read impact factor data and convert to nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    return img

def get_feature_data(tif_paths, window_size):
    """
    Read features and  min-max normalization
    """
    n = int(window_size/2)
    data = []
    for tif_data in os.listdir(tif_paths):
        img = read_data_from_tif(os.path.join(tif_paths, tif_data))
        img = (img-img.min())/(img.max()-img.min())
        w, h = img.shape[0], img.shape[1]
        print(str(tif_data)+'读取成功...')
        img = np.pad(img,(( n,n),(n,n)),'constant',constant_values = (0,0))  #Filling edges
        data.append(img)
    return w, h , len(data), np.array(data)  

def get_label_data(tif_path, window_size):
    """
    Read label data
    """
    n = int(window_size/2)
    img = read_data_from_tif(tif_path)
    img = np.pad(img,(( n,n),(n,n)),'constant',constant_values = (0.1,0.1)) #Filling 0.1
    return img

def shuffle_image_label(images, labels):
    """
    data shuffle
    """
    randnum = random.randint(0, len(images))
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels

def get_CNN_data(data, label, n):
    """
    Creating a CNN dataset
    """
    n = int(n/2)
    #Create train sets, labels
    train_data = []
    mask_0 = label == 0
    i_indices_0, j_indices_0 = np.where(mask_0)
    mask_1 = label == 2
    i_indices_1, j_indices_1 = np.where(mask_1)
    for i, j in zip(i_indices_0, j_indices_0):
        train_data.append((data[:,i-n:i+n+1,j-n:j+n+1],0))
    for i, j in zip(i_indices_1, j_indices_1):
        train_data.append((data[:,i-n:i+n+1,j-n:j+n+1],1))
    #Create validation sets, labels
    val_data = []
    mask_2 = label == 1
    i_indices_2, j_indices_2 = np.where(mask_2)
    mask_3 = label == 3
    i_indices_3, j_indices_3 = np.where(mask_3)
    for i, j in zip(i_indices_2, j_indices_2):
        val_data.append((data[:,i-n:i+n+1,j-n:j+n+1],0))
    for i, j in zip(i_indices_3, j_indices_3):
        val_data.append((data[:,i-n:i+n+1,j-n:j+n+1],1))
    
    train_imgs = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    val_imgs = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    train_imgs, train_labels = shuffle_image_label(train_imgs, train_labels)
    val_imgs, val_labels = shuffle_image_label(val_imgs, val_labels)
    
    train_imgs, val_imgs = np.array(train_imgs),np.array(val_imgs)
    train_labels, val_labels = np.array(train_labels).reshape((-1,1)), np.array(val_labels).reshape((-1,1))
    print(train_imgs.shape,val_imgs.shape, train_labels.shape,val_labels.shape )
    return train_imgs, train_labels, val_imgs, val_labels

def pixel_to_image(tif_paths, window_size):
    """
    Convert entire study area to data
    """
    imgs = []
    n = int(window_size/2)
    w, h ,n_feature, data = get_feature_data(tif_paths, window_size)
    for i in range(n, n+w):
        for j in range(n, n+h):
            imgs.append(data[:,i-n:i+n+1,j-n:j+n+1])
    return w, h ,n_feature, imgs

def generate_windows(data, slideWindow):
    """
    Step-by-step generation of window data and conversion to NumPy arrays
    """
    for i in range(0, len(data),slideWindow):
        window_data = data[i:i+slideWindow]
        window_np = np.array(window_data)
        yield window_np

def save_to_tif(data_path, pred_result):
    """
    save LSM
    """
    tif = gdal.Open(data_path)
    width, height = tif.RasterXSize, tif.RasterYSize

    img = pred_result.reshape(height, width)
    im_geotrans, im_prof = [], []  
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
    driver = gdal.GetDriverByName("GTiff")            
    dataset = driver.Create('Result/lsm_test.tif', im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans[-1])              #写入仿射变换参数
    dataset.SetProjection(im_prof[-1])                    #写入投影
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img)  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(img[i])
    del dataset
    print('ok')

def get_ML_data(tif_paths, label_path):
    """
    Creating a ML dataset
    """
    data = []
    data_name = []
    tif = gdal.Open(label_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    label = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    for tif_data in os.listdir(tif_paths):
        img = read_data_from_tif(os.path.join(tif_paths, tif_data))
        data_name.append(tif_data.split('.')[0])
        data.append(img)
    data_name.append('label')   
    data = np.array(data)
    train_data = []
    mask_0 = label == 0
    i_indices_0, j_indices_0 = np.where(mask_0)
    mask_1 = label == 2
    i_indices_1, j_indices_1 = np.where(mask_1)
    for i, j in zip(i_indices_0, j_indices_0):
        train_data.append((data[:,i,j],0))
    for i, j in zip(i_indices_1, j_indices_1):
        train_data.append((data[:,i,j],1))
    
    val_data = []
    mask_2 = label == 1
    i_indices_2, j_indices_2 = np.where(mask_2)
    mask_3 = label == 3
    i_indices_3, j_indices_3 = np.where(mask_3)
    for i, j in zip(i_indices_2, j_indices_2):
        val_data.append((data[:,i,j],0))
    for i, j in zip(i_indices_3, j_indices_3):
        val_data.append((data[:,i,j],1))
    
    train_imgs = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    val_imgs = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    train_imgs, train_labels = shuffle_image_label(train_imgs, train_labels)
    val_imgs, val_labels = shuffle_image_label(val_imgs, val_labels)
    
    train_imgs, val_imgs = np.array(train_imgs),np.array(val_imgs)
    train_labels, val_labels = np.array(train_labels).reshape((-1,1)), np.array(val_labels).reshape((-1,1))
    print(train_imgs.shape,val_imgs.shape, train_labels.shape,val_labels.shape )
    train_df = pd.DataFrame(np.concatenate((train_imgs, train_labels), axis=1), columns=data_name)
    val_df = pd.DataFrame(np.concatenate((val_imgs, val_labels), axis=1), columns=data_name)
    return train_df, val_df