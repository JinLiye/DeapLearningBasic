import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self): 
        super(SimpleCNN,self).__init__

        # nn.Sequential定义若干个序列，等价于依次调用
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            # 池化
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接层
        self.out = nn.Linear(32*7*7,10)

    def foorward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)

        return self.out(x)

class LeNet5(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet5,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5,1,0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )

        # self.fc = nn.Linear(256,120)
        # self.relu = nn.ReLU()

        # self.fc1 = nn.Linear(120,84)
        # self.relu1 = nn.ReLU()
        
        # self.fc2 = nn.Linear(84,num_classes)

        self.fc = nn.Sequential(
            nn.Linear(256,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
        




