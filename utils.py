import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

# # Construct SPP layer (Spatial Pyramid Pooling Layer)
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

#Fixed SPP-----> to be able to handle different sized inputs
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

# AUC plotting function
def drawAUC_TwoClass(y_true,y_score,path):
    fpr, tpr, thresholds =roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)  
    roc_auc = roc_auc*100
    plt.figure(figsize=(5,5),dpi=300)
    plt.plot(fpr, tpr, color='darkorange',linestyle='-',linewidth=2,label=(str(path).split('.')[0]+' = %0.2f %%)'% roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.tick_params(direction='in',top=True ,bottom=True,left=True,right=True)
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xticks(np.arange(0,1.1,0.1))
    
    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate') #fpr
    plt.ylabel('True Positive Rate')  #tpr
    plt.legend(loc="lower right")

    #print("AUC:",roc_auc)
    plt.savefig('Result/'+path, format='png')
    plt.cla()
    plt.close("all")

#plot loss
def draw_loss(loss1, loss2):
    plt.figure(figsize=(5,5,),dpi=300)
    plt.plot(range(len(loss1)),loss1, 'b', label='Training loss')    
    plt.plot(range(len(loss2)), loss2, 'r', label='validation loss')
    plt.tick_params(direction='in',top=True ,bottom=True,left=True,right=True)
    plt.xlabel('epochs') 
    plt.ylabel('loss')  
    plt.legend(loc='lower right')
    plt.savefig('Result/loss.png')
#plot acc
def draw_acc(acc1, acc2):
    plt.figure(figsize=(5,5,),dpi=300)
    plt.plot(range(len(acc1)),acc1, 'b', label='Training accuracy')    
    plt.plot(range(len(acc2)), acc2, 'r', label='validation accuracy')
    plt.tick_params(direction='in',top=True ,bottom=True,left=True,right=True)
    plt.xlabel('epochs') 
    plt.ylabel('accuracy')  
    plt.legend(loc='lower right')
    plt.savefig('Result/accuracy.png')
