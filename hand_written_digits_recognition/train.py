import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
from mlp import SimpleMLP

# 训练模型
def train(model,device,train_loader,optimizer,epoch,loss_func,writer:SummaryWriter = None):
    """模型训练函数
        这个函数的目的是通过训练数据集来训练神经网络模型。
        它遍历训练数据批次，执行前向传播、损失计算、反向传播和参数更新等操作，
        并周期性地输出训练进度信息。并运用tensorboard将训练过程可视化
    Args:
        model (torch.nn.Module): 要训练的神经网络模型
        device (torch.device): 指定的设备（CPU或GPU），用于模型计算
        train_loader (torch.utils.data.DataLoader): 用于加载训练数据的数据加载器
        optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数
        epoch (int): 当前的训练轮次
        loss_func (torch.nn.Module): 损失函数，用于计算模型预测与真实标签之间的损失
        writer(torch.utils.tensorboard.SummaryWritter):用于记录模型训练过程

    Returns:
        double: 本次训练的平均损失
    """

    # 将模型设置为训练模式
    model.train()

    # 记录本轮训练的平均loss
    train_loss = 0

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
        train_loss += loss.item()
        # 进行一次反向传播，计算模型梯度
        loss.backward()
        
        # 更新模型参数，优化器的一个步骤
        optimizer.step()
        
        # 写入tensorboard
        writer.add_scalar(f'loss_epoch_{epoch}',loss.item(),batch_idx)

        # 每处理完100个批次，输出训练进度信息
        if batch_idx % 100 == 0:
            # print(f"训练轮次：{epoch}[{batch_idx*len(data)}/{len(train_loader.dataset)}] 损失：{loss.item():.6f}")
            print(f"训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] 损失: {loss.item():.6f}")
    return train_loss / len(train_loader)
#  测试模型
def test(model,device,test_loader,epoch,loss_func,writer:SummaryWriter=None):
    """评估模型性能的函数

    Args:
        model (torch.nn.Module): 要评估的神经网络模型
        device (torch.device): 指定的设备（CPU或GPU），用于模型计算
        test_loader (torch.utils.data.DataLoader): 用于加载测试数据的数据加载器
        epoch(int): 当前的测试轮次
        loss_func (torch.nn.Module): 损失函数，用于计算模型预测与真实标签之间的损失
        writer(torch.utils.tensorboard.SummaryWritter):用于记录模型测试过程中的参数
    """

    # 将模型设置为评估模式
    model.eval()
    
    # 初始化测试损失
    test_loss = 0
    correct = 0

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
    
    for i in range(10):
        mask = (pred.view(-1) == i)
        if mask.sum() > 0:
            writer.add_images(f'epoch_{epoch},num={i}',data[mask])

    accuracy = correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print(f"\n测试集：平均损失：{test_loss:.4f},准确率：{correct}/{len(test_loader.dataset)}({100.*correct/len(test_loader.dataset):.0f}%)\n")
    
    return {'loss':test_loss,'accuracy':accuracy}

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

    # tensorboard
    writer = SummaryWriter(comment="_train")

    # 记录模型结构
    writer.add_graph(model,input_to_model=torch.rand(1,28,28))

    weight_save_dir = current_dir_path / 'weights'
    weight_save_dir.mkdir(exist_ok=True)

    # 运行训练和测试
    for epoch in range(1, 100):  # 总共训练5轮
        train_loss = train(model, device, train_loader, optimizer, epoch, loss_func,writer)

        # 写入本轮的loss
        writer.add_scalar('train_loss',train_loss,epoch)

        test_data = test(model, device, test_loader,epoch, loss_func,writer)

        # 写入本轮测试的loss
        writer.add_scalar('test_loss',test_data['loss'],epoch)
        writer.add_scalar('test_accuracy',test_data['accuracy'],epoch)

        # 保存模型训练数据
        torch.save(model.state_dict(),weight_save_dir / f'{int(time.time())}_epoch_{epoch}.pt')
    writer.close()

if __name__ == '__main__':
    main()