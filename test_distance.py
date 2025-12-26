import os
import numpy as np

import glob
from PIL import Image

from arcface import Arcface
from dizhi import find_images_in_folder

#还没有转128特征向量
def natural_sort_key(s):
    """Turn a string into a list of string and number chunks:
    "z23a" -> ["z", 23, "a"]
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


import re

def main(path1, path2):
    model = Arcface()
    print(len(path1),len(path2))

    distance_threshold = 1.17
    counter = 0
    counter1=0
    max=0
    sum=0
    for i in range(len(path1)):
        output1=Image.open(path1[i])
        output2=Image.open(path2[i])
        probability = model.detect_image(output1, output2)

        print(probability)
        sum=probability+sum
        if probability>max:
            max=probability
        if probability> distance_threshold:
            counter += 1

    print(f"距离大于 {distance_threshold} 的图片对数量: {counter}")
    print(f"距离大于 {1.79} 的图片对数量: {max}")
    print(sum)
    print(sum/len(path1))
#1.3897

if __name__ == "__main__":
    folder_path1 = 'result/outyanma3/test/greedyfool/adv'  # 标干文件
    imagesA = sorted(
        [os.path.join(folder_path1, f) for f in os.listdir(folder_path1) if f.lower().endswith(('png', 'jpg', 'jpeg'))],
        key=lambda x: natural_sort_key(os.path.basename(x))
    )
    print(imagesA[0:11])

    folder_path2 = 'attack2000/fill_112'  # 标干文件
    original_image_path2 = find_images_in_folder(folder_path2)
    print(original_image_path2[0:11])
    main(imagesA, original_image_path2)