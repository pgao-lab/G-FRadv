import glob

import torch
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time

start_time=time.time()
# 假设 data_folder 和 label_folder 是两个包含图片的文件夹
data_folder = "F-Img"
label_folder = "Atru1"

# 获取所有图片的文件路径
data_image_files = glob.glob(os.path.join(data_folder, "*.*"))  # 支持多种图片格式
label_image_files = glob.glob(os.path.join(label_folder, "*.*"))  # 支持多种图片格式

# 确保数据和标签的数量相匹配
assert len(data_image_files) == len(label_image_files), "The number of data images and label images must match."


# 定义一个加载器函数
def default_loader(path):
    return Image.open(path)


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])


class ImageToImageDataset(Dataset):
    def __init__(self, data_files, label_files, transform=None, loader=default_loader):
        self.data_files = data_files
        self.label_files = label_files
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        data_path = self.data_files[index]
        label_path = self.label_files[index]

        data_image = self.loader(data_path)
        label_image = self.loader(label_path)

        if self.transform is not None:
            data_image = self.transform(data_image)
            label_image = self.transform(label_image)  # 注意：对于标签图像，您可能不需要某些转换（如ToTensor后的归一化）

        # 返回数据和标签图像（都是张量）
        return data_image, label_image

    def __len__(self):
        return len(self.data_files)


# 创建数据集和数据加载器
dataset = ImageToImageDataset(data_image_files, label_image_files, transform=transform)
data_loader = DataLoader(dataset, batch_size=10, shuffle=False)
save_dir = 'saved_images'
os.makedirs(save_dir, exist_ok=True)
# 现在您可以使用 data_loader 来迭代加载数据和标签图像对
epoch=0
for data, labels in data_loader:
    # data 是输入图像批次，labels 是对应的标签图像批次
    # 注意：这里的 'labels' 实际上也是图像，而不是传统的类别标签
    epoch=epoch+1
    x=torch.norm(data - labels, p=2)
    print(data.shape, labels.shape)
    print(x)
    for img_idx, image_tensor in enumerate(data):
        # 将张量从[C, H, W]转换为[H, W, C]以符合PIL的输入要求
        image_pil = transforms.ToPILImage()(image_tensor)
        # 构造保存图像的路径和文件名
        save_path = os.path.join(save_dir, f'epoch_{epoch}img_{img_idx}.png')
        # 保存图像
        image_pil.save(save_path)
    # 进行您的训练/评估逻辑...

endtime=time.time()
print(start_time-endtime)