import torch
# 该模块提供了在PyTorch中构建神经网络的支持。
# 它包含用于定义各种层（如卷积层、循环层、线性层）
# 和神经网络训练中常用的损失函数的类和函数。
import torch.nn as nn

# 该模块包含各种函数，可以用于定义神经网络中的操作，
# 如激活函数（例如ReLU、sigmoid）、池化操作和损失函数（例如交叉熵损失）。
import torch.nn.functional as F

# 该模块提供了用于训练神经网络的各种优化算法的实现。
# 它包括诸如随机梯度下降（SGD）、Adam、RMSprop等优化器
import torch.optim as optim

# 这是一个专门设计用于计算机视觉任务的PyTorch库。
# 它提供了加载标准数据集、图像转换和常用预训练模型的实用程序。
from torchvision import datasets,transforms

# 这是PyTorch的一个实用工具，用于高效地加载数据。
# 它提供了一个可迭代对象，允许在训练期间轻松访问数据批次。
from torch.utils.data import DataLoader

# 这是Python标准库pathlib模块中的一个类。
# 它表示文件系统路径，并提供了用于文件系统相关操作的方法。
# 在这里，它用于以跨平台的方式处理文件路径。
from pathlib import Path

# 定义一个多层感知机的模型，该类继承自nn.module,这个基类提供了参数管理、梯度计算等功能
class SimpleMLP(nn.Module):
    """
    构建一个多层感知机类

    Attributes:
        fc1: 全连接层1
        fc2: 全连接层2
        fc3: 全连接层3
    """
    # 构造函数，在这里我们定义了网络的结构
    def __init__(self):
        """
        类的构造函数：初始化对象。
        """
        # 基类构造函数
        super(SimpleMLP,self).__init__()
        
        # 下面构建了三层全连接层，通过类nn.Linear创建
        # 第一个全连接层接受784的输入，这三MNIST图像展平后的大小（28*28），并输出大小为256的向量
        self.fc1 = nn.Linear(784,256)

        # 第二个全连接层接受256的输入，并输出大小为128的向量
        self.fc2 = nn.Linear(256,128)

        # 第一个全连接层接受128的输入，并输出大小为10的向量
        self.fc3 = nn.Linear(128,10)
        
    # 这是类的前向传播函数。在PyTorch中，
    # 每个nn.Module子类都必须实现forward方法，它定义了从输入到输出的数据流。
    # 在这个方法中，我们描述了数据如何通过网络层流动。
    def forward(self,x):
        """重载前向传播函数

        Args:
            x (_type_): 神经网络的输入

        Returns:
            _type_: 神经网络的输出
        """
        # 这一行代码将输入张量展平为一个一维张量
        # torch.flatten()函数用于将输入张量沿着指定的维度展平，
        # 第二个参数1表示从第二个维度（索引为1的维度，即通道维度）开始展平。
        x = torch.flatten(x,1)
        
        # 这一行代码将展平后的输入张量通过第一个全连接层self.fc1，然后通过ReLU激活函数进行非线性变换
        x = F.relu(self.fc1(x))
        
        # 这一行代码将通过第二个全连接层self.fc2，然后再次通过ReLU激活函数进行非线性变换。
        x = F.relu(self.fc2(x))
        
        # 第三层全连接输出
        x = self.fc3(x)

        # 最后，返回输出张量x，这就完成了前向传播过程。
        # 在使用nn.CrossEntropyLoss时，不需要在这里应用Softmax
        return x
    

# 训练模型
def train(model,device,train_loader,optimizer,epoch,loss_func):
    """模型训练函数
        这个函数的目的是通过训练数据集来训练神经网络模型。
        它遍历训练数据批次，执行前向传播、损失计算、反向传播和参数更新等操作，
        并周期性地输出训练进度信息。
    Args:
        model (torch.nn.Module): 要训练的神经网络模型
        device (torch.device): 指定的设备（CPU或GPU），用于模型计算
        train_loader (torch.utils.data.DataLoader): 用于加载训练数据的数据加载器
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数
        epoch (int): 当前的训练轮次
        loss_func (torch.nn.Module): 损失函数，用于计算模型预测与真实标签之间的损失
    """

    # 将模型设置为训练模式
    model.train()

    # enumerate()函数用于将一个可迭代对象（在这里是train_loader）转换为一个枚举对象，同时返回元素的索引和值。在每次迭代中，
    # batch_idx会是当前批次的索引，而(data, target)会是当前批次的数据和标签。
    for batch_idx,(data,target) in enumerate(train_loader):
        # 将数据迁移到制定的设备上进行运算
        data,target = data.to(device),target.to(device)
        
        # 梯度归零，因为pytorch会累积梯度
        optimizer.zero_grad()

        # 通过模型进行前向传播
        output = model(data)
        
        # 调用损失函数计算损失
        loss = loss_func(output,target)
        
        # 进行一次反向传播，计算模型梯度
        loss.backward()
        
        # 更新模型参数，优化器的一个步骤
        optimizer.step()
        
        # 每处理完100个批次，输出训练进度信息
        if batch_idx % 100 == 0:
            # print(f"训练轮次：{epoch}[{batch_idx*len(data)}/{len(train_loader.dataset)}] 损失：{loss.item():.6f}")
            print(f"训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
# 测试模型
def test(model,device,test_loader,loss_func):
    """评估模型性能的函数

    Args:
        model (torch.nn.Module): 要评估的神经网络模型
        device (torch.device): 指定的设备（CPU或GPU），用于模型计算
        test_loader (torch.utils.data.DataLoader): 用于加载测试数据的数据加载器
        loss_func (torch.nn.Module): 损失函数，用于计算模型预测与真实标签之间的损失
    """

    # 将模型设置为评估模式
    model.eval()
    
    # 初始化测试损失
    test_loss = 0

    # 初始化正确分类的样本数量
    correct = 0
    
    # 在评估阶段不需要进行梯度计算
    with torch.no_grad():
        # 遍历测试数据批次
        for data,target in test_loader:
            # 将数据移动到指定设备
            data,target = data.to(device),target.to(device)

            # 通过模型进行前向传播
            output = model(data)

            # 计算测试损失
            # .item() 方法用于获取一个张量的标量值，因为损失函数计算出的损失值通常是一个张量
            test_loss += loss_func(output,target).item()
            
            # 获取预测结果
            # .argmax(dim=1, keepdim=True): 这是一个张量操作，
            # 它的作用是沿着指定的维度（dim=1 表示第二维度，即类别维度）
            # 找到张量中每行最大值所在的索引。在分类任务中，通常选择得分最高的类别作为预测结果。
            # 因此，对于每个样本，argmax(dim=1) 将返回得分最高的类别的索引。
            # keepdim=True 表示保持结果张量的维度与原始张量相同。
            pred = output.argmax(dim = 1,keepdim = True)

            """
            pred.eq(target.view_as(pred)): 这是一个张量操作，它用来比较模型预测的类别 (pred) 
            和真实的类别 (target) 是否相等。.eq() 方法会逐元素比较两个张量，并返回一个布尔类型的张量，
            其中元素为 True 表示相等，False 表示不相等。
            
            target.view_as(pred) 的作用是将 target 张量的形状调整成和 pred 张量相同的形状，以便进行比较。

            .sum(): 这是一个张量方法，用来计算张量中所有元素的和。
            在这里，我们希望得到预测正确的样本数量，因此需要统计相等的元素的数量。

            .item(): 最后，.item() 方法用来获取一个张量的标量值，因为 sum() 方法返回的是一个张量，
            而我们通常只关心其数值，而不需要保留其在张量形式下的其他信息。
            """
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\n测试集：平均损失：{test_loss:.4f},准确率：{correct}/{len(test_loader.dataset)}({100.*correct/len(test_loader.dataset):.0f}%)\n")

def main():
    """主函数
    """
    # 设置设备，如果cuda不可用则用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取当前脚本所在的文件路径
    current_dir_path = Path(__file__).resolve().parent
    
    # 加载训练集，参数：数据集的本地路径、使用训练集还是测试集、不下载数据集、数据预处理流程（将图像转化为张量）
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
    # 创建了训练集和测试集的数据加载器，用于按批次加载数据。
    # 参数包括数据集对象、批次大小以及是否在每个 epoch 之后对数据进行重新排序。
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 实例化模型并移至设备
    model = SimpleMLP().to(device)

    # 定义损失函数，这里使用交叉熵
    loss_func = nn.CrossEntropyLoss()

    # 优化器，负责优化网络参数，这里使用了随机梯度下降（SGD）优化算法，并传入模型参数和学习率作为参数。
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 运行训练和测试
    for epoch in range(1, 6):  # 总共训练5轮
        train(model, device, train_loader, optimizer, epoch, loss_func)
        test(model, device, test_loader, loss_func)
    

if __name__ == '__main__':
    main()