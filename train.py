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
from logger import Logger
from sklearn.metrics import roc_curve, auc, confusion_matrix,f1_score
import pylab as plt
import numpy as np
import os
import csv
import math
import cv2

torch.cuda.set_device(0)


#读取多波段tif图像，将其转换为ndarray
def Myloader(path):
    dataset = gdal.Open(path)  # 读取栅格数据
    #print('处理图像的栅格波段数总共有：', dataset.RasterCount)

     # 判断是否读取到数据 
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出

    # 直接读取dataset,除0是将其转换为浮点类型
    img_array = dataset.ReadAsArray()/1.0
    #print(img_array)
    return img_array


# 得到一个包含路径与标签的列表
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data

#重写dataset
class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        #将图片转化为tensor
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

#得到对应图像的标签
def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'N' or str[i] == 'L') and str[i - 1] == '/':
            first = i
            break
    name = str[first:last]
    if name == 'Landslide':
        return 1
    else:
        return 0


#将数据送入模型
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        
        transforms.Normalize(mean=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
         std=(1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0))  # 归一化
    ])
        
    
    #数据路径
    path1 = 'cnn_datas2/training_data/Landslide/Landslide.%d.tif'
    path2 = 'cnn_datas2/training_data/NoLandslide/NoLandslide.%d.tif'

    path3 = 'cnn_datas2/testing_data/Landslide/Landslide.%d.tif'
    path4 = 'cnn_datas2/testing_data/NoLandslide/NoLandslide.%d.tif'

    data1= init_process(path1, [0, 516]) 
    data2 = init_process(path2, [0, 516])
    data3 = init_process(path3, [516, 737])
    data4 = init_process(path4, [516,737])
    #print(data2)

    #580个 训练
    train_data = data2[0:100]+data2[0:100]

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    #146个 测试
    test_data = data3[0:100]+data4[0:50]
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    #print(train_data)

    train_data = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=0)
    
    
    return train_data, test_data

def drawAUC_train(y_true,y_score):
    fpr, tpr, thresholds =roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    roc_auc = roc_auc * 100
    #开始画ROC曲线
    plt.figure(figsize=(5,5),dpi=300)
    plt.plot(fpr, tpr, color='darkorange',linestyle=':',linewidth=4,label='CNN (AUC = %0.2f%%)'% roc_auc)
    plt.legend(loc='lower right')#设定图例的位置，右下角
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.tick_params(direction='in',top=True ,bottom=True,left=True,right=True)#坐标轴朝向
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xticks(np.arange(0,1.1,0.1))
    
    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.legend(loc="lower right")
    if os.path.exists('./Train_AUC')==False:
        os.makedirs('./Train_AUC')
    plt.savefig('Train_AUC/train_aucx.png', format='png')

def drawAUC_TwoClass(y_true,y_score):
    fpr, tpr, thresholds =roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    roc_auc=roc_auc*100
    #开始画ROC曲线
    plt.figure(figsize=(5,5),dpi=300)
    plt.plot(fpr, tpr, color='darkorange',linestyle=':',linewidth=4,label='CNN (AUC = %0.2f %%)'% roc_auc)
    plt.legend(loc='lower right')#设定图例的位置，右下角
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.tick_params(direction='in',top=True ,bottom=True,left=True,right=True)#坐标轴朝向
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xticks(np.arange(0,1.1,0.1))
    
    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.legend(loc="lower right")
    if os.path.exists('./resultphoto')==False:
        os.makedirs('./resultphoto')
    print("AUC:",roc_auc)
    plt.savefig('resultphoto/AUC_TwoClassx.png', format='png')

def save_log(data1,data2):
    m=zip(data1,data2) 
    with open('D:/CNN/study/train.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for line in m:
             writer.writerow(line)
#定义softmax

#注意力机制
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

class cnn_std(nn.Module):
    def __init__(self):
        super(cnn_std, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=156,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=156),
            nn.MaxPool2d(kernel_size=2),
        )
        self.dropout=nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=156 * 4 * 4, out_features=16)
        self.fc2 = nn.Linear(in_features=16,out_features=2)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.conv2(x)
       
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        x=self.fc1(x)
        output = self.fc2(x)
        return output, x

class cnn_s(nn.Module):
    def __init__(self):
        super(cnn_s, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=64,
                kernel_size=3,
                #stride=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=156,
                kernel_size=3,
                #stride=1,
                #padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=156),
            nn.MaxPool2d(kernel_size=2),
        )
        self.dropout=nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=156 * 2 * 2, out_features=16)
        self.fc2 = nn.Linear(in_features=16,out_features=2)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.conv2(x)
       
        x = x.view(x.size(0), -1)


        x=self.fc1(x)
        output = self.fc2(x)
        x = self.dropout(x)
        return output, x

class cnn_s1(nn.Module):
    def __init__(self):
        super(cnn_s1, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
        )
        self.se1=SELayer(channel=64, reduction=16)
        self.pool1=nn.MaxPool2d(kernel_size=2)
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
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
        )
        self.se2=SELayer(channel=128,reduction=16)
        self.pool2=nn.MaxPool2d(kernel_size=2)
        self.se3=SELayer(channel=256,reduction=16)
        self.dropout=nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=256 * 2 * 2, out_features=2)
        #self.fc2 = nn.Linear(in_features=120, out_features=64)
        #self.fc3 = nn.Linear(in_features=64,out_features=2)

    def forward(self, x):

        x = x.type(torch.cuda.FloatTensor)

        x = self.conv1(x)
        x=self.se1(x)
        x=self.pool1(x)
        x = self.conv2(x)
        x=self.se2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.se3(x)
        x=self.pool2(x)

        
        x = x.view(x.size(0), -1)
        x=self.dropout(x)
        output=self.fc1(x)

        return output, x

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
        self.se1=SELayer(channel=64, reduction=16)
        
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
        self.se2=SELayer(channel=128, reduction=16)
        self.conv3=nn.Sequential(
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
        self.dropout=nn.Dropout2d(p=0.5)       
        self.fc1 = nn.Linear(in_features=196 * 4 * 4, out_features=384)
        self.fc2=nn.Linear(in_features=384,out_features=2)
        
    def forward(self, x):

        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x=self.se1(x)
        x = self.conv2(x)
        x=self.se2(x)
        x = self.conv3(x)

        #x=self.dropout(x)
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        output=self.fc2(x)

        return output, x




#训练及相关超参数设置
def train():
    train_log_path=r"./log/train_log"
    train_logger=Logger(train_log_path)
    train_loader, test_loader = load_data()
    epoch_num =250
    # GPU计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn_mul().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss().to(device)
    output_list=[]
    labels_list=[]    
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            
            optimizer.zero_grad()  # 梯度清0
            output = model(data)[0]  # 前向传播s
            _, preds = torch.max(output.data, 1)
            _, argmax = torch.max(output, 1)

            output_list.extend(output.detach().cpu().numpy())
            labels_list.extend(target.cpu().numpy())

            accuracy = (target == argmax.squeeze()).float().mean()

            loss = criterion(output, target)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            #计算一个epoch的值
            running_loss +=loss.item()*data.size(0)
            #计算一个epoch的准确率
            running_corrects +=torch.sum(preds==target.data)
            # 计算Loss和准确率的均值
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = float(running_corrects) / len(train_loader.dataset)
        #scheduler.step()
        print('{} Loss: {:.4f} Acc: {:.4f} Acc1:{:.4f}'.format('train', loss.item(), epoch_acc,accuracy)) 
        info = {'loss': epoch_loss, 'accuracy': epoch_acc}
        for tag, value in info.items():
             train_logger.scalar_summary(tag, value, epoch)
    score_array=np.array(output_list)
    drawAUC_train(labels_list,score_array[:,1])
    save_log(labels_list,score_array[:,1])
            

            

    torch.save(model, 'cnn1.pkl')

def test():
    train_loader, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('cnn1.pkl')  # load model
    #print(model)
    total = 0
    current = 0
    outputs_list = []     # 存储预测得分
    labels_list = []  
    TP=0
    TN=0
    FN=0
    FP=0
    for data in test_loader:
        #model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        outputs_list.extend(outputs.detach().cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

        predicted = torch.max(outputs.data, 1)[1].data       
       
        # TP    predict 和 label 同时为1
        TP += ((predicted == 1) & (labels.data == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN += ((predicted == 0) & (labels.data == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN += ((predicted == 0) & (labels.data == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP += ((predicted == 1) & (labels.data == 0)).cpu().sum()
        total += labels.size(0)
        current += (predicted == labels).sum()


    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1_score=(2*precision*recall)/(precision+recall) 
    MCC=(TP*TN-FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    print("precision:",precision)   
    print("recall:",recall)   
    print("F1_score:",F1_score)   
    print("MCC:",MCC)
    print('Accuracy: %d %%' % (100 * current / total))
    score_array=np.array(outputs_list)
    drawAUC_TwoClass(labels_list,score_array[:,1])
    #save_log(labels_list,score_array[:,1])

def predicted_tif():
    train_loader, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('cnn1.pkl')  # load model
    total = 0
    current = 0
    outputs_list = []     # 存储预测得分
    labels_list = []  
    for data in test_loader:
        #model.eval() 
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        outputs_list.extend(outputs.detach().cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

        predicted = torch.max(outputs.data, 1)[1].data    
        outputs=outputs.data.cpu().numpy()
        result=softmax(outputs)
        print(result)
        print(predicted)
        total += labels.size(0)
        current += (predicted == labels).sum()
    print('Accuracy: %d %%' % (100 * current / total))

def save_log(data1,data2):
    m=zip(data1,data2) 
    with open('D:/CNN/study/ROC85.csv', 'w', newline='') as csvfile:
        writer  = csv.writer(csvfile)
        for line in m:
             writer.writerow(line)


def img_preprocess():
    #读取多波段tif图像，将其转换为ndarray

    #dataset = gdal.Open("D:/CNN/cnn_datas2/testing_data/Landslide/Landslide.580.tif")
    dataset = gdal.Open("D:/CNN/cnn_datas2/training_data/NoLandslide/NoLandslide.500.tif")  # 读取栅格数据
    #print('处理图像的栅格波段数总共有：', dataset.RasterCount)

     # 判断是否读取到数据 
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出

    # 直接读取dataset,除0是将其转换为浮点类型
    img_array = dataset.ReadAsArray()/1.0
    print(img_array.shape)

    
    transform = transforms.Compose([
        transforms.ToTensor(),      
        transforms.Normalize(mean=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
         std=(1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0))  # 归一化
    ])
    img_array=transform(img_array)
    img_array = img_array.unsqueeze(0)	
    #print(img_array.size())
    return img_array




# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)



# 计算grad-cam并可视化
def cam_show_img(feature_map, grads):
    #H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (17, 17))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img =  heatmap 

    #path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite("cam1.jpg", cam_img)
    return heatmap

if __name__=='__main__':
    #train()
    #test()
    #predicted_tif()
    

     # 存放梯度和特征图
    fmap_block = list()
    grad_block = list() 
    #导入图像
    img_input=img_preprocess()
    
    #加载训练好的pth文件
    #pthfile = './cnn.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.load('cnn1.pkl')
    print(net)
    #net.eval()														# 8
    net.conv3[-1].register_forward_hook(farward_hook)	# 9
    net.conv3[-1].register_backward_hook(backward_hook)

        # forward
    img_input=img_input.to(device)
    output = net(img_input)[0]
    idx = np.argmax(output.cpu().data.numpy())
    predicted = torch.max(output.data, 1)[1].data  
    print(idx)
    print(predicted)

        # backward
    net.zero_grad()
    class_loss = output[0,idx]
    class_loss.backward()
    
    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    #cam_show_img(fmap, grads_val)



