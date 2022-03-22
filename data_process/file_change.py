#导入库函数
import os
import shutil

#images文件夹路径
path1='D:/桌面/test'
#待分类文件路径
path2='D:/桌面/test/images'

data=[]


#获取待分类图片名称，存入data数组
for root, dirs, files in os.walk(path1):    
    data=files
print(data)

#获取待分类文件夹名称

for  dirs in os.listdir(path1):
    #print(dirs) #当前路径下所有子目录
    #进行字符串匹配，若相同则放入指定文件夹
    for i in range(len(data)): 
        temp=data[i].split("dog")[0]
        if temp==dirs:

            print("1")
        else:
            print("0")



        