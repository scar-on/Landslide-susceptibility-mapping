import torch 
import read_data
import numpy as np
from torch.autograd import Variable
from model import LSM_cnn
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
import config
config=config.config


def save_LSM():
    print('*******************************************生成LSM*******************************************')
    model = LSM_cnn(config["feature"])
    model.load_state_dict(torch.load("Result/best.pth"))
    model.to(config["device"])
    tensor_data = read_data.get_feature_data()
    print('整个预测区域大小：'+str(tensor_data.shape))

    creat = read_data.creat_dataset(tensor_data, config["size"])
    data = creat.creat_new_tensor()
    images_list = []
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(config["width"]):
            for j in range(config["height"]):
                images_list.append(data[:, i:i+creat.n, j:j+creat.n].astype(np.float32))
                if((i!=0 and i%config["Cutting_window"]==0 and j==config["height"]-1) or (i==config["width"]-1 and j==config["height"]-1)):
                    print('i='+str(i)+' j='+str(j))
                    pred_data = np.stack(images_list)
                    images_list=[]
                    pred_dataset = TensorDataset(torch.from_numpy(pred_data))
                    pred_loader = DataLoader(dataset=pred_dataset,batch_size=config["batch_size"], shuffle=False)

                    for images in tqdm(pred_loader):
                        images= torch.tensor([item.cpu().detach().numpy() for item in images]).cuda()
                        images = Variable(images.squeeze(0)).to(config["device"])
                        probs.append(model(images)[:,0].cpu().numpy())
    probs = np.concatenate(probs)
    print('概率列表生成完成')
    read_data.save_to_tif(probs, 'Result/lsm_test.tif')

