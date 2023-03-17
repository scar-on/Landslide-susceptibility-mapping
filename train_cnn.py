import os
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch import optim
import numpy as np
from model import LSM_cnn
from utils import drawAUC_TwoClass
from torch.autograd import Variable
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='Result/log_dir')
import config
config=config.config


# train函数
def train(alldata_train, alltarget_train, alldata_val, alltarget_val):
    max_acc=0
    train_dataset = TensorDataset(torch.from_numpy(alldata_train).float(),torch.from_numpy(alltarget_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=config["batch_size"], shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(alldata_val).float(),torch.from_numpy(alltarget_val).float())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=config["batch_size"], shuffle=True)

    model=LSM_cnn(config["feature"]).to(config["device"])
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss().to(config["device"])
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])  #Lr=0.01
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 
    
    for epoch in range(config["epochs"]):
        scheduler.step()  # 更新学习率
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        train_outputs_list = []     
        train_labels_list = [] 
        val_outputs_list = []     
        val_labels_list = [] 

        model.train()
        for images, target in train_loader:
            #反向传播
            images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
            optimizer.zero_grad()
            outputs = model(images)
            _,preds = torch.max(outputs.data,1)
            loss = criterion(outputs, target.squeeze().long()) 
            loss.backward()  
            optimizer.step()  

            train_outputs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(target.cpu().numpy())
            train_array=np.array(train_outputs_list)
            
            train_acc += (preds[..., None]==target).squeeze().sum().cpu().numpy()
            train_loss += loss.item()

        writer.add_scalars('LOSS/',  {'Train_Loss':train_loss/len(train_dataset)}, epoch)
        writer.add_scalars('ACC/', {'Train_Acc':float(train_acc)/ len(train_dataset)}, epoch)


        model.eval()
        with torch.no_grad():
            for  images, target in val_loader:
                images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
                outputs = model(images)
                loss = criterion(outputs, target.squeeze().long())
                val_loss += loss.item()
                
                val_outputs_list.extend(outputs.detach().cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())
                score_array=np.array(val_outputs_list)
                _, preds = torch.max(outputs.data, 1)
                val_acc += (preds[..., None]==target).squeeze().sum().cpu().numpy()

            print('[%03d/%03d]  Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, config["epochs"], \
               train_acc / len(train_dataset), train_loss /len(train_dataset), val_acc / len(val_dataset),
               val_loss /len(val_dataset)))
            if  (val_acc / len(val_dataset)) > max_acc:
                max_acc = val_acc / len(val_dataset)
                drawAUC_TwoClass(val_labels_list, score_array[:,1], 'val_AUC.png')
                drawAUC_TwoClass(train_labels_list, train_array[:,1], 'train_AUC.png')
                torch.save(model.state_dict(), 'Result/best.pth')
            # 记录Loss, accuracy
            writer.add_scalars('LOSS/valid', {'valid_loss': val_loss /len(val_dataset)}, epoch)
            writer.add_scalars('ACC/valid', {'valid_acc': val_acc / len(val_dataset)}, epoch)
    torch.save(model.state_dict(), 'Result/latest.pth')