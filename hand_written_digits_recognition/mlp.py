import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from pathlib import Path

class SimpleMLP(nn.Module):
    def __init__(self):
        # 基类构造函数
        super(SimpleMLP,self).__init__()
        
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)
        
    def forward(self,x):
        x = torch.flatten(x,1)
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x
    

# 训练模型
def train(model,device,train_loader,optimizer,epoch,loss_func):
    model.train()
    
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        
        optimizer.zero_grad()

        output = model(data)
        
        loss = loss_func(output,target)
        
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % 100 == 0:
            # print(f"训练轮次：{epoch}[{batch_idx*len(data)}/{len(train_loader.dataset)}] 损失：{loss.item():.6f}")
            print(f"训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
# 测试模型
def test(model,device,test_loader,loss_func):
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += loss_func(output,target).item()
            
            pred = output.argmax(dim = 1,keepdim = True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\n测试集：平均损失：{test_loss:.4f},准确率：{correct}/{len(test_loader.dataset)}({100.*correct/len(test_loader.dataset):.0f}%)\n")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    current_dir_path = Path(__file__).resolve().parent
    
    # 加载训练集，参数：数据集的本地路径、使用训练集还是测试集、不下载数据集、数据预处理流程
    train_dataset = datasets.MNIST(root=str(current_dir_path / 'data'), 
                                   train=True, 
                                   download=False, 
                                   transform=transforms.ToTensor())
    
    # 加载测试集，参数含义同上
    test_dataset = datasets.MNIST(root=str(current_dir_path / 'data'), 
                                  train=False, 
                                  download=False, 
                                  transform=transforms.ToTensor())
    # 从数据集创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 实例化模型并移至设备
    model = SimpleMLP().to(device)

    # 定义损失函数，这里使用交叉熵
    loss_func = nn.CrossEntropyLoss()

    # 优化器，负责优化网络参数，这里使用SGD算法
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 运行训练和测试
    for epoch in range(1, 6):  # 总共训练5轮
        train(model, device, train_loader, optimizer, epoch, loss_func)
        test(model, device, test_loader, loss_func)
    

if __name__ == '__main__':
    main()








