import os
import glob

import torchvision
from PIL import Image


def find_images_in_folder(folder_path):
    """
    查找指定文件夹内所有.jpg和.png格式的图片，并将它们的路径存储到数组中。

    :param folder_path: 包含图片的文件夹路径
    :return: 包含所有图片路径的数组
    """
    # 初始化一个空列表来存储图片路径
    image_paths = []

    # 使用glob模块查找.jpg和.png文件
    # os.path.join用于确保路径的正确拼接，特别是在不同操作系统上
    jpg_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    png_files = glob.glob(os.path.join(folder_path, '*.png'))

    # 将找到的文件路径添加到列表中
    image_paths.extend(jpg_files)
    image_paths.extend(png_files)

    # 如果需要，也可以添加对其他图片格式的支持
    # 例如，添加对.jpeg, .gif等格式的支持

    return image_paths


# 示例用法
# folder_path = 'huiduronghe/yangben_256_160x4'  # 修改为你的图片文件夹路径
# image_paths = find_images_in_folder(folder_path)
# print(image_paths)
#
# for i in range(3):
#     x = Image.open(image_paths[i]).convert('L')
#     trans = torchvision.transforms.ToTensor()
#     x = trans(x)
#     print(x.shape)

