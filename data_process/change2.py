import os
import shutil

#待处理数据路径
path='D:/CNN/cnn_datas/testing_data/NoLandslide'
#
path1='D:/CNN/cnn_datas2/testing_data/NoLandslide'

#1. 首先挑出后缀为tif的文件，单独存入一个文件夹
#2. 将挑出的tif文件进行重新命名 ，例如00000000.tif------>>Landslide.0.tif

i=516
for filename in os.listdir(path):
    #print(filename)
    temp=filename.split(".")[0]
    if temp=='NoLandslide':
        print(temp)
        newname="NoLandslide."+str(i)+".tif"
        os.rename(path+'/'+filename,path1+'/'+newname)
        i+=1