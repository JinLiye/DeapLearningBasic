import torch
from pathlib import Path
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from mlp import SimpleMLP

# 权重文件
weight_file = '1708328479_epoch_5.pt'

# 获取权重文件路径
current_dir_path = Path(__file__).resolve().parent
model_weights_path = current_dir_path / 'weights' / weight_file

# 输入图片路径
imput_image_dir = current_dir_path / '..'/ 'my_num'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_input_data(input_image_dir,img_size):
    # list,存储所有输入图片的路径
    imgs_path = input_image_dir.glob('*.png')
    # print(input_image_dir)
    # for i in imgs_path:
    #     print(i)

    # 转换图片流水线
    transform = transforms.Compose([
        # 灰度图
        transforms.Grayscale(),
        # 修改尺寸
        transforms.Resize(img_size),
    ])

    # 读取图片并预处理
    input_imgs = []
    for img_path in imgs_path:
        # 读取图片
        img = read_image(str(img_path))

        # transform
        img = transform(img)

        # 归一
        img = img.type(torch.float32)/255

        # 灰度翻转
        img = img * -1 + 1

        input_imgs.append(img)
    # (n, 1, 28, 28)
    data = torch.stack(input_imgs).to(device)
    return data

def main():
    data = get_input_data(imput_image_dir,(28,28))

    writer = SummaryWriter(comment=f'_predict_{weight_file}')

    model = SimpleMLP().to(device)

    model.load_state_dict(torch.load(model_weights_path))

    model.eval()

    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim = 1,keepdim = True)

    for i in range(10):
        mask = (pred.view(-1) == i)
        if mask.sum() > 0:
            writer.add_images(f'num={i}',data[mask])

if __name__ == '__main__':
    main()







