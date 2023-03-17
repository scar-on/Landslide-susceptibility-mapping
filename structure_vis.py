import torch.nn as nn
from torchsummary import summary
from utils import Modified_SPPLayer  



class LSM_cnn(nn.Module):
    def __init__(self, in_chanel):
        super(LSM_cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chanel, 64, 3, 1, 1),  # [64, 128, 128]
            nn.ReLU(),
            nn.BatchNorm2d(64),            
            nn.MaxPool2d(2),  # [64, 64, 64]
            )
        self.conv2=nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # [128, 32, 32]
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
            
        
        #self.fc1 = nn.Linear(256*4*4, 384)
        self.fc1 = nn.Linear(1280, 384)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(384, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        spp = Modified_SPPLayer(2).forward(x)
        #x = x.view(x.size()[0], -1)
        
        x = self.fc1(spp)
        x = self.dropout(x)
        out= self.fc2(x)
        return out


if __name__=='__main__':
    print("程序开始执行---------")
    model = LSM_cnn(4).cuda()
    print(model)
    print(summary(model, (4, 25, 25),device='cuda'))