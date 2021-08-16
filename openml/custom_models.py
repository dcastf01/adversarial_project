import torch.nn as nn
import torch.nn.functional as F
import torch
class AlexNet(nn.Module): 
    #se cambiaron las capas fc por el mlp
    def __init__(self, in_chans,width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential( # Input 1*28*28
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1), # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2), # 32*14*14
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*7*7
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 128*7*7
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 256*7*7
            )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2), # 256*3*3
            nn.ReLU(inplace=True),
            )
        self.fc1 = nn.Linear(256*3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256*3*3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

class Inception(nn.Module):
    def __init__(self,in_channel,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.norm1_1=nn.BatchNorm2d(in_channel,eps=1e-3)
        self.p1_1=nn.Conv2d(in_channels=in_channel,out_channels=c1,kernel_size=1)
        self.norm2_1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p2_1=nn.Conv2d(in_channels=in_channel,out_channels=c2[0],kernel_size=1)
        self.norm2_2 = nn.BatchNorm2d(c2[0], eps=1e-3)
        self.p2_2=nn.Conv2d(in_channels=c2[0],out_channels=c2[1],kernel_size=3,padding=1)
        self.norm3_1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p3_1=nn.Conv2d(in_channels=in_channel,out_channels=c3[0],kernel_size=1)
        self.norm3_2 = nn.BatchNorm2d(c3[0], eps=1e-3)
        self.p3_2=nn.Conv2d(in_channels=c3[0],out_channels=c3[1],kernel_size=5,padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.norm4_2 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.p4_2 = nn.Conv2d(in_channels=in_channel, out_channels=c4, kernel_size=1)
 
    def forward(self, x):
        p1=self.p1_1(F.relu(self.norm1_1(x)))
        p2=self.p2_2(F.relu(self.norm2_2(self.p2_1(F.relu(self.norm2_1(x))))))
        p3=self.p3_2(F.relu(self.norm3_2(self.p3_1(F.relu(self.norm3_1(x))))))
        p4=self.p4_2(F.relu(self.norm4_2(self.p4_1(x))))
        return torch.cat((p1,p2,p3,p4),dim=1)
 
#Test Inception block
# test_net = Inception(3, 64, (48, 64), (64, 96), 32)
# test_x = Variable(torch.zeros(1, 3, 96, 96))
# print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
# test_y = test_net(test_x)
# print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
 
 
 
class GoogleNet(nn.Module):
    def __init__(self,in_chans=1,num_classes=10):
        super(GoogleNet,self).__init__()
        layers=[]
        layers+=[nn.Conv2d(in_channels=in_chans,out_channels=64,kernel_size=7,stride=2,padding=3),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        layers+=[nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
                 nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        layers+=[Inception(192,64,(96,128),(16,32),32),
                 Inception(256,128,(128,192),(32,96),64),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1)]
        layers+=[Inception(480, 192, (96, 208), (16, 48), 64),
                 Inception(512, 160, (112, 224), (24, 64), 64),
                 Inception(512, 128, (128, 256), (24, 64), 64),
                 Inception(512, 112, (144, 288), (32, 64), 64),
                 Inception(528, 256, (160, 320), (32, 128), 128),
               nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        layers += [Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                #    nn.AvgPool2d(kernel_size=2)
                   ]
        self.net = nn.Sequential(*layers)
        self.dense=nn.Linear(1024,num_classes)
 
 
    def forward(self,x):
        x=self.net(x)
        x=x.view(-1,1024*1*1)
        x=self.dense(x)
        return x