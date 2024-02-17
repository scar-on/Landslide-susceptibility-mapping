import json
import torch
import argparse
import data_prepare as dp
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset

from model import LSM_cnn
from torch import optim
from utils import drawAUC_TwoClass, draw_acc, draw_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN Processes on data")
    parser.add_argument( "--feature_path", default='origin_data/feature/', type=str)
    parser.add_argument( "--label_path", default='origin_data/label/label1.tif', type=str)
    parser.add_argument( "--window_size", default=15, type=int)
    parser.add_argument( "--lr", default=0.0001, type=float)
    parser.add_argument( "--batch_size", default=128, type=int)
    parser.add_argument( "--epochs", default=300, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    _, _, n_feature, data = dp.get_feature_data(args.feature_path, args.window_size)
    label = dp.get_label_data(args.label_path, args.window_size)
    alldata_train, alltarget_train, alldata_val, alltarget_val = dp.get_CNN_data(data, label, args.window_size)

    train_dataset = TensorDataset(torch.from_numpy(alldata_train).float(),torch.from_numpy(alltarget_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(alldata_val).float(),torch.from_numpy(alltarget_val).float())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=args.batch_size, shuffle=True)


    model=LSM_cnn(n_feature).to('cuda')
    # Loss Functions and Optimizers
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(model.parameters(), lr = args.lr) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     
    max_acc=0
    record = {"train": {"acc": [], "loss": []}, "val": {"acc": [], "loss": []}}
    for epoch in range(args.epochs):
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
            #backward 
            images, target = images.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            _,preds = torch.max(outputs.data,1)
            loss = criterion(outputs, target.squeeze().long()) 
            loss.backward()  
            optimizer.step()  

            train_outputs_list.extend(outputs.detach().cpu().numpy()) 
            train_array=np.array(train_outputs_list)
            train_labels_list.extend(target.cpu().numpy())

            
            train_acc += (preds[..., None]==target).squeeze().sum().cpu().numpy()
            train_loss += loss.item()

        record["train"]["loss"].append(train_loss/len(train_dataset))
        record["train"]["acc"].append(train_acc/ len(train_dataset))

        model.eval()
        with torch.no_grad():
            for  images, target in val_loader:
                images, target = images.to("cuda"), target.to("cuda")
                outputs = model(images)
                loss = criterion(outputs, target.squeeze().long())
                val_loss += loss.item()
                
                val_outputs_list.extend(outputs.detach().cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())
                score_array=np.array(val_outputs_list)
                _, preds = torch.max(outputs.data, 1)
                val_acc += (preds[..., None]==target).squeeze().sum().cpu().numpy()

            print('[%03d/%03d]  Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
                (epoch + 1, args.epochs, \
                train_acc / len(train_dataset), train_loss /len(train_dataset), val_acc / len(val_dataset),
                val_loss /len(val_dataset)))
            if  (val_acc / len(val_dataset)) > max_acc:
                max_acc = val_acc / len(val_dataset)

                drawAUC_TwoClass(val_labels_list, score_array[:,1], 'val_AUC.png')
                drawAUC_TwoClass(train_labels_list, train_array[:,1], 'train_AUC.png')
                torch.save(model.state_dict(), 'Result/best.pth')
            # Record Loss, accuracy
            record["val"]["loss"].append(val_loss /len(val_dataset))
            record["val"]["acc"].append(val_acc / len(val_dataset))
    scheduler.step()    
    draw_acc(record["train"]["acc"], record["val"]["acc"])
    draw_loss(record["train"]["loss"], record["val"]["loss"])

    with open('Result/record.json', 'w') as f:
        json.dump(record, f)
    torch.save(model.state_dict(), 'Result/latest.pth')

if __name__=='__main__':
    main()