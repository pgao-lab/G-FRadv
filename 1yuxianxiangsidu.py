# import os
# import torch
# import numpy as np
# from torch import nn
#
#
# def process_files_in_subfolders(main_folder_path):
#     """
#     遍历主文件夹中的所有子文件夹，每个子文件夹包含两个文件，
#     读取这两个文件作为Tensor，计算欧氏距离
#
#     参数:
#     main_folder_path: 主文件夹路径
#     """
#     # 确保主文件夹存在
#     if not os.path.exists(main_folder_path):
#         print(f"错误: 文件夹 '{main_folder_path}' 不存在")
#         return
#
#     # 获取所有子文件夹
#     subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]
#
#     if not subfolders:
#         print(f"在 '{main_folder_path}' 中没有找到子文件夹")
#         return
#
#     print(f"找到 {len(subfolders)} 个子文件夹")
#
#     # 遍历每个子文件夹
#     for i, subfolder in enumerate(subfolders):
#         print(f"\n处理子文件夹 {i + 1}/{len(subfolders)}: {os.path.basename(subfolder)}")
#
#         # 获取子文件夹中的所有文件
#         files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
#
#         # 确保只有两个文件
#         if len(files) != 2:
#             print(f"警告: 子文件夹 '{os.path.basename(subfolder)}' 中包含 {len(files)} 个文件，而不是2个")
#             continue
#
#         # 按文件名排序以确保一致性
#         files.sort()
#
#         # 构建文件路径
#         file1_path = os.path.join(subfolder, files[0])
#         file2_path = os.path.join(subfolder, files[1])
#
#         try:
#             # 读取文件
#             tensor1 = torch.load(file1_path)
#             tensor2 = torch.load(file2_path)
#
#             # 检查Tensor形状
#             if tensor1.shape != (1, 12, 1280) or tensor2.shape != (1, 12, 1280):
#                 print(f"警告: 文件形状不符合预期。期望 (1, 12, 1280)，实际: {tensor1.shape} 和 {tensor2.shape}")
#                 continue
#
#             # 重塑Tensor
#             tensor1_reshaped = tensor1.squeeze(0)
#             tensor2_reshaped = tensor2.squeeze(0)
#             cosim = nn.CosineSimilarity(dim=1)
#             # 分别计算眼睛、鼻子、嘴巴的余弦相似度，然后求平均
#             loss_lip =cosim(tensor1_reshaped[1:2], tensor2_reshaped[1:2])
#             loss_eyebrows =cosim(tensor1_reshaped[2:3], tensor2_reshaped[2:3])
#             loss_eye = cosim(tensor1_reshaped[3:4], tensor2_reshaped[3:4])
#             loss_hair =cosim(tensor1_reshaped[4:5], tensor2_reshaped[4:5])
#             loss_nose = cosim(tensor1_reshaped[5:6], tensor2_reshaped[5:6])
#             loss_skin = cosim(tensor1_reshaped[6:7], tensor2_reshaped[6:7])
#
#
#
#             # print(loss_eyebrows, loss_hair, loss_skin,loss_lip,loss_eye,loss_nose)
#             #
#             # loss = (loss_eye + loss_nose + loss_lip+loss_eyebrows+loss_skin+loss_hair) / 6.0
#             # # 打印结果
#             # print(f"文件1: {files[0]}, 文件2: {files[1]}")
#             # print(f"欧氏距离矩阵形状: {loss.shape}")
#             # print(f"行间欧氏距离均值: {loss}")
#         except Exception as e:
#             print(f"处理文件时出错: {e}")
#             continue
#
#
# # 使用示例
# if __name__ == "__main__":
#     main_folder = "attack2000/outcome"  # 替换为你的主文件夹路径
#     process_files_in_subfolders(main_folder)

import os
import torch
import torch.nn as nn
import numpy as np


def process_files_in_subfolders(main_folder_path):
    """
    遍历主文件夹中的所有子文件夹，每个子文件夹包含两个文件，
    读取这两个文件作为Tensor，计算欧氏距离

    参数:
    main_folder_path: 主文件夹路径

    返回:
    all_similarities: 形状为(n, 6)的张量，n是子文件夹数量
    """
    # 确保主文件夹存在
    if not os.path.exists(main_folder_path):
        print(f"错误: 文件夹 '{main_folder_path}' 不存在")
        return None

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]

    if not subfolders:
        print(f"在 '{main_folder_path}' 中没有找到子文件夹")
        return None

    print(f"找到 {len(subfolders)} 个子文件夹")

    # 初始化一个列表来存储所有相似度结果
    all_similarities = []

    # 遍历每个子文件夹
    for i, subfolder in enumerate(subfolders):
        print(f"\n处理子文件夹 {i + 1}/{len(subfolders)}: {os.path.basename(subfolder)}")

        # 获取子文件夹中的所有文件
        files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]

        # 确保只有两个文件
        if len(files) != 2:
            print(f"警告: 子文件夹 '{os.path.basename(subfolder)}' 中包含 {len(files)} 个文件，而不是2个")
            continue

        # 按文件名排序以确保一致性
        files.sort()

        # 构建文件路径
        file1_path = os.path.join(subfolder, files[0])
        file2_path = os.path.join(subfolder, files[1])

        try:
            # 读取文件
            tensor1 = torch.load(file1_path)
            tensor2 = torch.load(file2_path)

            # 检查Tensor形状
            if tensor1.shape != (1, 12, 1280) or tensor2.shape != (1, 12, 1280):
                print(f"警告: 文件形状不符合预期。期望 (1, 12, 1280)，实际: {tensor1.shape} 和 {tensor2.shape}")
                continue

            # 重塑Tensor
            tensor1_reshaped = tensor1.squeeze(0)
            tensor2_reshaped = tensor2.squeeze(0)
            cosim = nn.CosineSimilarity(dim=1)

            # 分别计算眼睛、鼻子、嘴巴的余弦相似度
            loss_lip = cosim(tensor1_reshaped[1:2], tensor2_reshaped[1:2]).item()
            loss_eyebrows = cosim(tensor1_reshaped[2:3], tensor2_reshaped[2:3]).item()
            loss_eye = cosim(tensor1_reshaped[3:4], tensor2_reshaped[3:4]).item()
            loss_hair = cosim(tensor1_reshaped[4:5], tensor2_reshaped[4:5]).item()
            loss_nose = cosim(tensor1_reshaped[5:6], tensor2_reshaped[5:6]).item()
            loss_skin = cosim(tensor1_reshaped[6:7], tensor2_reshaped[6:7]).item()

            # 将当前子文件夹的6个相似度值添加到列表中
            subfolder_similarities = [loss_skin,0.5,loss_hair, loss_eyebrows, loss_nose, loss_eye, loss_lip]
            all_similarities.append(subfolder_similarities)

        except Exception as e:
            print(f"处理文件时出错: {e}")
            continue

    # 将列表转换为张量
    if all_similarities:
        all_similarities_tensor = torch.tensor(all_similarities)
        print(f"\n处理完成，得到形状为 {all_similarities_tensor.shape} 的张量")
        return all_similarities_tensor
    else:
        print("没有成功处理任何子文件夹")
        return None


# 使用示例
if __name__ == "__main__":
    main_folder = "attack2000/outcome"  # 替换为你的主文件夹路径
    result = process_files_in_subfolders(main_folder)

    if result is not None:
        # 保存结果到文件
        torch.save(result, "3yanma/similarities_results.pt")
        print(f"结果已保存到 similarities_results.pt")

        # 也可以保存为numpy格式或其他格式
        np.save("similarities_results.npy", result.numpy())
        print(f"结果已保存到 similarities_results.npy")