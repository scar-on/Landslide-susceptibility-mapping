## Landslide susceptibility mapping (LSM)

### 数据说明
- 影响因子与标签均为TIF格式，且尺寸宽高均相同
- 影响因子命名为 **xxx_5.tif** (5为重分类数量，可修改为自己数据的值)
- 标签命名需含有 **label** (便于程序自动识别)  见下图
- 标签需制作成具有0、1、2、3四类型的点，具体参考项目 **example_label.tif**

![image](https://user-images.githubusercontent.com/57258378/225853069-a1f1eefe-32d1-46ea-a1ea-13ae98c75581.png)

### 运行说明
- 将项目clone到与data文件夹（影响因子与标签均存放于data文件夹）同一目录下
- 将cofig中第一个参数修改为自己的data文件夹名称, 其余参数均有详细注释
- 运行 **start.py**
### 必要库安装
- GDAL   2.4.1

