import numpy as np
import torch


def process_mask(a, b):
    """
    处理掩码数据：将a中0-6的值替换为b中对应角标的值，-1替换为0.5

    参数:
    a: 形状为(112, 112)的数组，包含-1到6的整数
    b: 形状为(10, 7)的tensor，对应值0-6

    返回:
    形状为(10, 112, 112)的tensor，包含10个处理后的掩码
    """
    # 确保a是numpy数组，b是torch tensor
    if isinstance(a, torch.Tensor):
        a_np = a.numpy()
    else:
        a_np = np.array(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    # 初始化结果数组，形状为(10, 112, 112)
    result = torch.zeros((2500, 112, 112))

    # 对于b的每一行（共10行）
    for i in range(2500):
        # 创建一个与a形状相同的临时数组，初始值设为0.5
        temp_mask = torch.full((112, 112), 0.5)

        # 处理值为0-6的位置
        for value in range(7):  # 0到6
            # 找到a中等于value的位置
            indices = (a_np == value)
            # 将这些位置的值设置为b[i, value]
            temp_mask[indices] = b[i, value]

        # 将处理后的掩码添加到结果中
        result[i] = temp_mask

    return result


# 示例用法
if __name__ == "__main__":
    # 加载数据
    a_example = torch.load('output/color_regions_tensor.pt')
    print("a数据形状:", a_example.shape)

    b_example = torch.load('similarities_results.pt')
    print("b数据形状:", b_example.shape)

    # 处理数据
    result_masks = process_mask(a_example, b_example)
    torch.save(result_masks, "fuzhiyanma.pt")
    print("结果已保存到 fuzhiyanma.pt")

    print("结果掩码形状:", result_masks.shape)
    print(f"Tensor中的唯一数字: {torch.unique(result_masks[0])}")
    print("第一个结果掩码的示例值:")
    print(result_masks[0, :5, :5])  # 显示前5x5的值

    # 验证-1的值是否都变成了0.5
    print("\n验证-1的值是否都变成了0.5:")
    print("掩码中值为0.5的位置:", (result_masks[0] == 0.5).sum().item(), "个")
    print("原始掩码中-1的总数:", (a_example == -1).sum())