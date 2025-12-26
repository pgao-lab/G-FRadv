import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
import cv2
import os

# 设置matplotlib使用非交互式后端以避免兼容性问题
matplotlib.use('Agg')


def analyze_color_regions(image_path, num_colors=6):  # 修改为6种非绿色颜色
    """
    分析图像中的颜色区域并以tensor数据表示

    参数:
        image_path: 输入图像路径
        num_colors: 要识别的颜色数量(不包括绿色和白色)

    返回:
        color_tensor: 表示颜色区域的tensor，绿色为0，白色为-1，其他颜色为1-6
        color_centers: 识别出的颜色中心(RGB值)
        original_image: 原始图像对象
    """
    # 打开并转换图像
    original_image = Image.open(image_path).convert('RGB')
    width, height = original_image.size
    transform = transforms.ToTensor()
    image_tensor = transform(original_image)

    # 将图像转换为numpy数组以便处理
    img_array = np.array(original_image).reshape(-1, 3)

    # 识别绿色区域
    r, g, b = img_array[:, 0], img_array[:, 1], img_array[:, 2]
    # 绿色通道明显强于红色和蓝色通道的区域被认为是绿色
    green_mask = (g > r * 1.2) & (g > b * 1.2) & (g > 50)

    # 识别白色区域 (RGB值都接近255)
    white_mask = (r > 200) & (g > 200) & (b > 200)

    # 从非绿色和非白色区域中提取像素进行聚类
    target_pixels_mask = ~(green_mask | white_mask)
    target_pixels = img_array[target_pixels_mask]

    if len(target_pixels) > 0:
        # 使用K-means聚类识别主要颜色
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10)
        kmeans.fit(target_pixels)

        # 获取聚类中心和标签
        color_centers = kmeans.cluster_centers_.astype(int)
        target_labels = kmeans.labels_ + 1  # 从1开始编号

        # 创建完整的标签数组
        full_labels = np.zeros(len(img_array), dtype=int)
        full_labels[green_mask] = 0  # 绿色标记为0
        full_labels[white_mask] = -1  # 白色标记为-1
        full_labels[target_pixels_mask] = target_labels  # 其他颜色标记为1-n

        # 将标签数组重塑为图像形状
        label_map = full_labels.reshape(height, width)

        # 转换为tensor
        color_tensor = torch.from_numpy(label_map).float()

        return color_tensor, color_centers, original_image
    else:
        # 如果没有目标像素，返回全零tensor
        color_tensor = torch.zeros(height, width)
        return color_tensor, None, original_image


def visualize_color_analysis(original_image, color_tensor, color_centers, save_path=None):
    """
    可视化颜色分析结果

    参数:
        original_image: 原始图像
        color_tensor: 颜色区域tensor
        color_centers: 颜色中心列表
        save_path: 保存结果图像的路径
    """
    # 计算需要的行数和列数
    n_colors = len(color_centers) if color_centers is not None else 0
    n_plots = 2 + n_colors  # 原始图像 + 颜色分布 + 颜色块

    # 计算合适的网格布局
    n_cols = min(4, n_plots)  # 最多4列，以适应更多颜色
    n_rows = (n_plots + n_cols - 1) // n_cols  # 向上取整

    plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # 显示原始图像
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_image)
    plt.title("原始图像")
    plt.axis('off')

    # 显示颜色区域分布
    plt.subplot(n_rows, n_cols, 2)
    # 创建一个自定义颜色映射，将-1(白色)映射为白色，0(绿色)映射为绿色，其他颜色使用viridis
    cmap = plt.cm.viridis
    cmap.set_under('white')  # 值小于0的显示为白色
    cmap.set_over('green')  # 值大于最大值的显示为绿色
    plt.imshow(color_tensor, cmap=cmap, vmin=-0.5, vmax=6.5)
    plt.title("颜色区域分布\n(绿色=0, 白色=-1, 其他颜色=1-6)")
    plt.axis('off')

    # 创建颜色条
    if color_centers is not None:
        # 显示识别出的颜色
        for i, color in enumerate(color_centers):
            plt.subplot(n_rows, n_cols, 3 + i)
            color_block = np.ones((100, 100, 3), dtype=np.uint8)
            color_block[:, :] = color
            plt.imshow(color_block)
            plt.title(f"颜色 {i + 1}: {color}")
            plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"颜色分析结果已保存到: {save_path}")
    else:
        plt.show()


def extract_color_positions(color_tensor):
    """
    提取每种颜色的位置坐标

    参数:
        color_tensor: 颜色区域tensor

    返回:
        color_positions: 字典，包含每种颜色的位置坐标
    """
    color_positions = {}

    # 绿色区域的位置
    green_positions = torch.where(color_tensor == 0)
    color_positions["绿色"] = list(zip(green_positions[0].numpy(), green_positions[1].numpy()))

    # 白色区域的位置
    white_positions = torch.where(color_tensor == -1)
    color_positions["白色"] = list(zip(white_positions[0].numpy(), white_positions[1].numpy()))

    # 提取其他颜色的位置（最多6种非绿色非白色颜色）
    for i in range(1, 7):  # 修改为1-6
        positions = torch.where(color_tensor == i)
        if len(positions[0]) > 0:  # 只有在该颜色存在时才添加
            color_positions[f"颜色{i}"] = list(zip(positions[0].numpy(), positions[1].numpy()))

    return color_positions


def visualize_color_locations(original_image, color_tensor, color_index, color_name, save_path=None):
    """
    可视化特定颜色在图像中的位置

    参数:
        original_image: 原始图像
        color_tensor: 颜色区域tensor
        color_index: 颜色索引(-1,0-6)
        color_name: 颜色名称
        save_path: 保存结果图像的路径
    """
    # 创建图像副本
    img_array = np.array(original_image.copy())

    # 获取指定颜色的位置
    positions = torch.where(color_tensor == color_index)
    coords = list(zip(positions[1].numpy(), positions[0].numpy()))  # 注意坐标顺序(x,y)

    # 在图像上标记这些位置
    for x, y in coords:
        # 在指定位置绘制矩形标记
        cv2.rectangle(img_array, (x - 1, y - 1), (x + 1, y + 1), (255, 0, 0), 1)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_array)
    plt.title(f"{color_name} 在图像中的位置")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"{color_name}位置可视化已保存到: {save_path}")
    else:
        plt.show()


def analyze_color_stats(color_tensor, image_size):
    """
    分析颜色区域的统计信息

    参数:
        color_tensor: 颜色区域tensor
        image_size: 图像尺寸

    返回:
        stats: 每种颜色的统计信息字典
    """
    total_pixels = image_size[0] * image_size[1]
    stats = {}

    # 绿色区域统计
    green_count = torch.sum(color_tensor == 0).item()
    green_percentage = (green_count / total_pixels) * 100
    stats["绿色"] = {
        "像素数量": green_count,
        "占比(%)": round(green_percentage, 2)
    }

    # 白色区域统计
    white_count = torch.sum(color_tensor == -1).item()
    white_percentage = (white_count / total_pixels) * 100
    stats["白色"] = {
        "像素数量": white_count,
        "占比(%)": round(white_percentage, 2)
    }

    # 其他颜色区域统计（最多6种非绿色非白色颜色）
    for i in range(1, 7):  # 修改为1-6
        count = torch.sum(color_tensor == i).item()
        if count > 0:  # 只有在该颜色存在时才添加
            percentage = (count / total_pixels) * 100
            stats[f"颜色{i}"] = {
                "像素数量": count,
                "占比(%)": round(percentage, 2)
            }

    return stats


def create_color_masks(color_tensor, output_dir):
    """
    为每种颜色创建二值掩码并保存

    参数:
        color_tensor: 颜色区域tensor
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 为每种颜色创建掩码（最多6种非绿色非白色颜色）
    masks = {
        "绿色": (color_tensor == 0).float(),
        "白色": (color_tensor == -1).float(),
        "颜色1": (color_tensor == 1).float(),
        "颜色2": (color_tensor == 2).float(),
        "颜色3": (color_tensor == 3).float(),
        "颜色4": (color_tensor == 4).float(),
        "颜色5": (color_tensor == 5).float(),
        "颜色6": (color_tensor == 6).float()
    }

    # 保存每种颜色的掩码
    for color_name, mask in masks.items():
        if torch.sum(mask) > 0:  # 只有在该颜色存在时才保存
            mask_np = mask.numpy()
            plt.imsave(os.path.join(output_dir, f"{color_name}_mask.png"), mask_np, cmap='gray')

    print(f"颜色掩码已保存到目录: {output_dir}")


def main():
    # 图像路径
    image_path = "in_out/img2.png"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 分析颜色区域
    print("正在分析图像颜色区域...")
    color_tensor, color_centers, original_image = analyze_color_regions(image_path, num_colors=6)  # 指定6种非绿色非白色颜色

    # 输出基本结果
    print("颜色区域tensor形状:", color_tensor.shape)
    print("识别出的颜色中心(RGB):")
    if color_centers is not None:
        for i, color in enumerate(color_centers):
            print(f"颜色 {i + 1}: {color}")

    # 可视化颜色分析结果
    print("\n生成颜色分析可视化...")
    visualize_color_analysis(original_image, color_tensor, color_centers,
                             save_path=os.path.join(output_dir, "color_analysis.png"))

    # 提取颜色位置
    print("\n提取颜色位置信息...")
    color_positions = extract_color_positions(color_tensor)

    # 可视化每种颜色的位置
    print("生成颜色位置可视化...")
    for color_name, positions in color_positions.items():
        if color_name == "绿色":
            color_index = 0
        elif color_name == "白色":
            color_index = -1
        else:
            color_index = int(color_name.replace("颜色", ""))

        visualize_color_locations(
            original_image, color_tensor, color_index, color_name,
            save_path=os.path.join(output_dir, f"{color_name}_locations.png")
        )

    # 分析颜色统计信息
    print("\n计算颜色统计信息...")
    image_size = original_image.size
    stats = analyze_color_stats(color_tensor, image_size)

    print("颜色区域统计:")
    for color, info in stats.items():
        print(f"{color}: {info['像素数量']} 像素, 占比 {info['占比(%)']}%")

    # 创建颜色掩码
    print("\n创建颜色掩码...")
    create_color_masks(color_tensor, os.path.join(output_dir, "masks"))

    # 保存tensor数据
    print("\n保存Tensor数据...")
    torch.save(color_tensor, os.path.join(output_dir, 'color_regions_tensor.pt'))
    print("Tensor数据已保存为 'color_regions_tensor.pt'")

    print(f"\n所有结果已保存到目录: {output_dir}")


if __name__ == "__main__":
    main()