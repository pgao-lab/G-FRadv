import cv2
import numpy as np
from sklearn.cluster import KMeans


def quantize_colors(image_path, k=3, output_path="output_quantized.jpg"):
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查路径是否正确")
        return

    # 2. 将图像从BGR转换为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. 将图像数据重塑为二维数组（每个像素是一个RGB向量）
    pixels = image_rgb.reshape(-1, 3)

    # 4. 使用K-means聚类提取主要颜色
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_  # 每个像素的聚类标签
    dominant_colors = kmeans.cluster_centers_.astype(int)  # 主要颜色

    # 5. 创建一个新的图像，用主要颜色替换原始像素
    quantized_image = dominant_colors[labels].reshape(image_rgb.shape).astype(np.uint8)

    # 6. 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
    print(f"量化后的图像已保存到 {output_path}")


# 示例用法
input_image_path = "img.png"  # 输入图像路径
output_image_path = "output_quantized.jpg"  # 输出图像路径
quantize_colors(input_image_path, k=3, output_path=output_image_path)

