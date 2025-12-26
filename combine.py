import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import ToPILImage
from dizhi import find_images_in_folder

# 图像转换器
transform = transforms.Compose([
    transforms.ToTensor(),
])


def load_image(image_path):
    """加载并转换图像为张量"""
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def process_tensors(tensor1, tensor2, tensor3):
    """
    使用张量操作高效处理三个输入张量的融合
    相同符号情况使用0.3, 0.3, 0.4的加权平均
    不同符号情况:
        数值选择绝对值最大的值
        符号由投票机制决定（多数符号决定最终符号，平票时保留原始符号）
    """
    # 堆叠三个张量以便并行处理
    stacked = torch.stack([tensor1, tensor2, tensor3], dim=0)

    # 计算符号并检查是否所有符号相同
    signs = torch.sign(stacked)
    all_same = (signs[0] == signs[1]) & (signs[1] == signs[2])

    # 符号相同的情况：使用0.3, 0.3, 0.4的加权平均
    weights = torch.tensor([0.3, 0.3, 0.4], device=stacked.device).view(3, 1, 1)
    same_sign_result = (stacked * weights).sum(dim=0)

    # 符号不同的情况：数值取绝对值最大的值，符号由投票机制决定
    # 计算绝对值并找到每个位置的最大绝对值索引
    abs_values = torch.abs(stacked)
    max_abs_idx = torch.argmax(abs_values, dim=0, keepdim=True)

    # 获取绝对值最大的原始值（保留符号）
    max_abs_value = torch.gather(stacked, 0, max_abs_idx).squeeze(0)

    # 计算符号投票
    # 统计正号、负号和零的数量
    positive_count = (signs > 0).sum(dim=0).float()
    negative_count = (signs < 0).sum(dim=0).float()

    # 确定投票结果：
    # 1. 如果正号数量 > 负号数量，符号为正
    # 2. 如果负号数量 > 正号数量，符号为负
    # 3. 如果平票（包括全零情况），保留绝对值最大值的原始符号
    vote_sign = torch.where(
        positive_count > negative_count,
        torch.ones_like(max_abs_value),
        torch.where(
            negative_count > positive_count,
            -torch.ones_like(max_abs_value),
            torch.sign(max_abs_value)  # 平票时保留原始符号
        )
    )

    # 应用投票符号：保留绝对值最大值的幅度，但使用投票决定的符号
    diff_sign_result = torch.abs(max_abs_value) * vote_sign

    # 合并相同符号和不同符号的结果
    result = torch.where(all_same, same_sign_result, diff_sign_result)

    return result.unsqueeze(0)  # 添加通道维度

# 主处理流程
if __name__ == "__main__":
    # 输入输出路径配置
    original_dir = '../attack2000/112'
    adv_dirs = [
        '../result/outyanma2/test/greedyfool/adv',
        '../result/outyanma3/test/greedyfool/adv',
        '../result/outyanma4/test/greedyfool/adv'
    ]
    save_dir = '../ronghe/RH2500yanma1'

    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有图像路径
    orig_paths = find_images_in_folder(original_dir)
    adv_paths_list = [find_images_in_folder(d) for d in adv_dirs]

    # 验证文件数量一致性
    num_files = len(orig_paths)
    # for paths in adv_paths_list:
    #     if len(paths) != num_files:
    #         raise ValueError(f"对抗样本目录包含 {len(paths)} 个文件，但原始目录有 {num_files} 个文件")

    # 处理每张图像
    for i in range(num_files):
        print(f"处理图像 {i + 1}/{num_files}")

        try:
            # 加载原始图像和对抗样本
            orig_img = load_image(orig_paths[i])
            adv_imgs = [load_image(adv_paths[i]) for adv_paths in adv_paths_list]

            # 计算差异张量
            diffs = [adv - orig_img for adv in adv_imgs]

            # 分通道处理并融合差异
            channel_results = []
            for c in range(3):  # 对RGB三个通道分别处理
                channel_diffs = [diff[c, :, :] for diff in diffs]
                fused = process_tensors(*channel_diffs)
                channel_results.append(fused)

            # 合并通道并生成新图像
            fused_diff = torch.cat(channel_results, dim=0)
            new_img = orig_img + fused_diff
            new_img = torch.clamp(new_img, 0, 1)  # 确保像素值有效

            # 保存结果
            output_path = os.path.join(save_dir,  f'{i:05d}.png')
            ToPILImage()(new_img).save(output_path)

        except Exception as e:
            print(f"处理图像 {i} 时出错: {str(e)}")
            continue