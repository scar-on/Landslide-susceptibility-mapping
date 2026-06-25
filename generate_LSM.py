import torch 
import argparse
import numpy as np
from tqdm import tqdm
import data_prepare as dp
from model import LSM_cnn
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
    parser.add_argument( "--model_path", default='Result/best.pth', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('*******************************************generate LSM*******************************************')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    slideWindow = args.slide_window*args.slide_window
    _, _ , n_feature, data_list = dp.pixel_to_image(args.feature_path,args.window_size)
    model = LSM_cnn(n_feature)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    probs = []
    model.eval()
    with torch.no_grad():
        for window in dp.generate_windows(data_list, slideWindow):
                pred_dataset = TensorDataset(torch.from_numpy(window).float())
                pred_loader = DataLoader(dataset=pred_dataset,batch_size=args.batch_size, shuffle=False)
                for images in tqdm(pred_loader):
                    images = images[0].to(device)
                    probs.append(torch.softmax(model(images), dim=1)[:,1].cpu().numpy())
    probs = np.concatenate(probs)
    print('Finsih!')
    dp.save_to_tif(args.label_path, probs)

if __name__=='__main__':
    main()
