import torch 
import argparse
import numpy as np
from tqdm import tqdm
import data_prepare as dp
from model import LSM_cnn
from torch.autograd import Variable
from torch.utils.data import DataLoader,TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Gaussian Processes on MNIST")
    parser.add_argument( "--feature_path", default='origin_data/feature/', type=str)
    parser.add_argument( "--label_path", default='origin_data/label/label1.tif', type=str)
    parser.add_argument( "--window_size", default=15, type=int)
    parser.add_argument( "--lr", default=0.0001, type=float)
    parser.add_argument( "--batch_size", default=128, type=int)
    parser.add_argument( "--epochs", default=300, type=int)
    parser.add_argument( "--slide_window", default=512, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('*******************************************generate LSM*******************************************')
    model = LSM_cnn(9)
    model.load_state_dict(torch.load("Result/best.pth"))
    model.to('cuda')

    slideWindow = args.slide_window*args.slide_window
    _, _ , _, data_list = dp.pixel_to_image(args.feature_path,args.window_size)
    probs = []
    model.eval()
    with torch.no_grad():
        for window in dp.generate_windows(data_list, slideWindow):
                pred_dataset = TensorDataset(torch.from_numpy(window))
                pred_loader = DataLoader(dataset=pred_dataset,batch_size=128, shuffle=False)
                for images in tqdm(pred_loader):
                    images= torch.tensor(np.array([item.cpu().detach().numpy() for item in images])).cuda()
                    images = Variable(images.squeeze(0)).to('cuda')
                    probs.append(model(images)[:,0].cpu().numpy())
    probs = np.concatenate(probs)
    print('Finsih!')
    dp.save_to_tif(args.label_path, probs)

if __name__=='__main__':
    main()
