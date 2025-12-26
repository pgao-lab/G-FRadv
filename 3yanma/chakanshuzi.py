import torch
import os

# 假设你有一个保存的Tensor文件（.pt格式）
# 如果没有，我们先创建一个示例文件
if not os.path.exists('sample_tensor.pt'):
    sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    torch.save(sample_tensor, 'sample_tensor.pt')
    print("已创建示例Tensor文件: sample_tensor.pt")

# 加载Tensor数据
try:
    tensor_data = torch.load('fuzhiyanma.pt')
    print("成功加载Tensor数据:")
    print(tensor_data)

    # 分析Tensor数据
    print(f"\nTensor形状: {tensor_data.shape}")
    print(f"Tensor数据类型: {tensor_data.dtype}")
    print(f"Tensor中的唯一数字: {torch.unique(tensor_data[2])}")
    print(f"Tensor中的唯一数字: {len(tensor_data[0])}")
    print(f"Tensor中的最大值: {torch.max(tensor_data)}")
    print(f"Tensor中的最小值: {torch.min(tensor_data)}")
    print(f"Tensor中的平均值: {torch.mean(tensor_data.float())}")

    # 访问Tensor中的具体元素
    print(f"\n第一行数据: {tensor_data[0]}")
    print(f"第二列数据: {tensor_data[:, 1]}")
    print(f"位于(1,2)的元素: {tensor_data[1, 2]}")

except FileNotFoundError:
    print("错误: 未找到Tensor文件")
except Exception as e:
    print(f"加载Tensor时出错: {e}")