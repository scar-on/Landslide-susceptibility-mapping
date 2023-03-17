import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type
    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size() 
        for i in range(self.num_levels):
            level = i+1
            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten

#修正后的SPP-----> 能够使其处理不同大小的输入
class Modified_SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        #super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type


    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size() 
#         print(x.size())
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1],pooling[1],pooling[0],pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))


            # 选择池化方式 
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)



            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return  x_flatten

# AUC绘制函数
def drawAUC_TwoClass(y_true,y_score,path):
    fpr, tpr, thresholds =roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    roc_auc=roc_auc*100
    #开始画ROC曲线
    plt.figure(figsize=(5,5),dpi=300)
    plt.plot(fpr, tpr, color='darkorange',linestyle='-',linewidth=2,label=('CNN ('+str(path).split('.')[0]+' = %0.2f %%)'% roc_auc))
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

    #print("AUC:",roc_auc)
    plt.savefig('Result/'+path, format='png')
