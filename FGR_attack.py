import time
import time
import torch.nn.functional as F
import torchvision

from dizhi import find_images_in_folder
from nets.arcface import Arcface
from options import BaseOptions
import os
import sys
from data_loader import McDataset
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn

from predict_advval import adc

from utils.utils import (get_num_classes, seed_everything, show_config,
                         worker_init_fn)


from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torchvision.models as models
import numpy as np
import random
from torchvision.transforms import ToPILImage
import torch.nn as nn

from shutil import copyfile
from piqa import SSIM
cudnn.benchmark = True

pool_kernel = 3
Avg_pool = nn.AvgPool2d(pool_kernel, stride=1, padding=int(pool_kernel / 2))


def main():
    opt = BaseOptions().parse()
    print(torch.cuda.device_count())

    # --------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # --------------------------------------#
    cuda = True
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    # --------------------------------------#
    backbone = "mobilefacenet"
    # --------------------------------------#
    #   输入图像大小
    # --------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    model_path = "model_data/arcface_mobilefacenet.pth"

    model = Arcface(backbone=backbone, mode="")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    netT = model.eval()
    import numpy as np

    def tv_loss(r):
        """
        输入 r: 4D 张量 (Batch x Channel x Height x Width)，可位于 CPU 或 GPU
        """
        # 计算水平方向差异（下方像素差）
        h_diff = r[:, :, :-1, :-1] - r[:, :, 1:, :-1]
        # 计算垂直方向差异（右侧像素差）
        v_diff = r[:, :, :-1, :-1] - r[:, :, :-1, 1:]
        # 合并差异并求和（全程使用 PyTorch 操作，无需转 NumPy）
        loss = torch.sqrt(h_diff ** 2 + v_diff ** 2+1e-8).sum()
        return loss

    if cuda:
        netT = torch.nn.DataParallel(netT)
        cudnn.benchmark = True
        netT = netT.cuda()



    im_size = 112
    test_dataset = McDataset(
        opt.dataroot,
        transform=transforms.Compose([
            transforms.ToTensor(),

        ]),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(test_loader))

    file_root = os.path.join('result/test/file/', opt.phase, opt.name)
    if not os.path.exists(file_root):
        os.makedirs(file_root)



    root = os.path.join('result/out/', opt.phase, opt.name)
    eps = opt.max_epsilon * 2 / 255.
    print(eps)
    Iter = int(opt.iter)
    print("Iter {0}".format(Iter))
    print("EPS {0}".format(opt.max_epsilon))

    folder_path = 'attack2000/shizhen/x4'  # 失真图片文件夹路径
    image_paths = find_images_in_folder(folder_path)
    print(len(image_paths))

    folder_path1='attack2000/fill_112'#最近标干文件
    image_paths1 = find_images_in_folder(folder_path1)
    print(len(image_paths1))

    folder_path2 = 'attack2000/112'#原始图片按
    image_paths2 = find_images_in_folder(folder_path2)
    print(len(image_paths2))

    ssim_loss = SSIM()
    ssim_loss = ssim_loss.cuda()
    loss_m = nn.MSELoss()
    loss_m = loss_m.cuda()
    Baccu = []
    for i in range(1):
        temp_accu = AverageMeter()
        Baccu.append(temp_accu)


    if opt.max_epsilon >= 16:
        boost = False
    else:
        boost = True
    # print("Boost:{0}".format(boost))
    yanma = torch.load('3yanma/fuzhiyanma.pt')
    yanma=yanma.cuda()
    print(yanma.shape)
    for idx, data in enumerate(test_loader):
        print(idx, '++++++++++++++++++++++++++')


        input_A = data['A']
        # print(type(input_A),'000000000000000')
        input_A = input_A.cuda()
        # torchvision.utils.save_image(input_A,'z.png')
        real_A = Variable(input_A, requires_grad=False)
        image_names = data['name']
        x = Image.open(image_paths[idx]).convert('L')
        x1=Image.open((image_paths1[idx]))
        x2 = Image.open(image_paths2[idx]).convert('L')
        trans = torchvision.transforms.ToTensor()
        x = trans(x)
        x = x.cuda()
        image_hill = x
        x1=trans(x1)
        x1=x1.cuda()
        x1=x1.view(1,3,112,112)
        x2 = trans(x2)
        x2 = x2.cuda()
        x2 = x2.view(1, 1, 112, 112)
        t1=x-x2

        t=t1.view(1, 1, -1)

        # t2=torch.abs(t)
        pre_hill = 1-image_hill

        pre_hill = pre_hill.view(1, 1, -1)

        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 0)
        # print(percen,'++++++++++++')
        pre_hill = torch.max(pre_hill - percen, torch.zeros(pre_hill.size()).cuda())
        # print(pre_hill.shape, '33333333333333333333')
        np_hill = pre_hill.detach().cpu().numpy()
        percen = np.percentile(np_hill, 70)
        pre_hill /= percen


        pre_hill = torch.clamp(pre_hill, 0, 1)
        pre_hill = Avg_pool(pre_hill)
        # print(pre_hill.shape, '55555555555555555')
        SIZE = int(im_size * im_size)
        a_yuan = netT(real_A, "predict")
        b_yuan=netT(x1,'predict')

        _, target = torch.max(a_yuan, 1)

        adv = real_A
        ini_num = 100
        grad_num = ini_num
        mask = torch.zeros(1, 3, SIZE).cuda()
        block_size = 2  # 分块大小（可调，如8x8或16x16）
        k = t1 # 扩展为[1,1,112,112]以适配unfold操作

        # 分块操作：将k划分为block_size x block_size的块
        k_blocks = F.unfold(k, kernel_size=block_size, stride=block_size)  # [1, block_size*block_size, num_blocks]
        k_blocks = k_blocks.view(1, 1, block_size, block_size, -1)  # [1, 1, block_size, block_size, num_blocks]

        # 对每个块计算重要性
        k_block_importance = k_blocks.mean(dim=(2, 3), keepdim=True)  # [1,1,1,1,num_blocks]

        # 将重要性扩展回原始块形状
        k_processed_blocks = k_block_importance.expand(1, 1, block_size, block_size,
                                                       -1)  # [1,1,block_size,block_size,num_blocks]
        k_processed_blocks = k_processed_blocks.reshape(1, block_size * block_size, -1)  # [1,block_size^2,num_blocks]

        # 合并块回原始图像形状
        k_processed = F.fold(k_processed_blocks, output_size=(112, 112), kernel_size=block_size, stride=block_size)
        k_processed = k_processed.squeeze()
        k_processed = (k_processed - k_processed.min()) / (k_processed.max() - k_processed.min() + 1e-8)  # 归一化到[0,1]
        k_processed = k_processed * 20
        # temp_eps = eps/8
        temp_eps = eps / 4 * yanma[idx].view(112, 112)

        ##### begin
        for iters in range(40):

            temp_A = Variable(adv.data, requires_grad=True)
            logist_B_A = netT(temp_A, "predict")
            loss_pixel = loss_m(temp_A,real_A)
            loss_ssim = 1 - ssim_loss(temp_A, real_A)
            if torch.norm(logist_B_A-b_yuan,p=2)>1.79:
                break
            Loss = torch.norm(logist_B_A-b_yuan,p=2)+0.1*loss_pixel+0.3*loss_ssim
            # print(torch.norm(logist_B_A-b_yuan,p=2))
            netT.zero_grad()
            if temp_A.grad is not None:
                temp_A.grad.data.fill_(0)
            Loss.backward(retain_graph=True)
            grad = temp_A.grad
            k_weight = k_processed.view(1, 1, 112, 112).expand_as(grad)  # [1,3,112,112]
            weighted_grad = grad * k_weight

            abs_grad = torch.abs(weighted_grad).view(1, 3, -1).mean(1, keepdim=True)

            abs_grad = abs_grad * pre_hill

            if not boost:
                abs_grad = abs_grad * (1 - mask)

            _, grad_sort_idx = torch.sort(abs_grad)
            grad_sort_idx = grad_sort_idx.view(-1)

            grad_idx = grad_sort_idx[-grad_num:]

            mask[0, :, grad_idx] = 1.
            temp_mask = mask.view(1, 3, im_size, im_size)
            grad = temp_mask * grad
            abs_grad = torch.abs(grad)
            abs_grad = abs_grad / torch.max(abs_grad)
            normalized_grad = abs_grad * grad.sign()
            scaled_grad = normalized_grad.mul(temp_eps)*5.0
            temp_A = temp_A + scaled_grad
            temp_A = clip(temp_A, real_A, eps)
            adv = torch.clamp(temp_A, 0, 1)
            if boost:
                grad_num += ini_num

        final_adv = adv
        adv_noise = real_A - final_adv
        adv = final_adv

        abs_noise = torch.abs(adv_noise).view(1, 3, -1).mean(1, keepdim=True)

        temp_mask = abs_noise != 0

        modi_num = torch.sum(temp_mask).data.clone().item()


        adv_noise = real_A - adv
        # torchvision.utils.save_image(adv, 'result/test/file/2.png')
        abs_noise = torch.abs(adv_noise).view(1, 3, -1).mean(1, keepdim=True)
        if not os.path.exists(root):
            os.makedirs(root)
        if not os.path.exists(os.path.join(root, 'clean')):
            os.makedirs(os.path.join(root, 'clean'))
        if not os.path.exists(os.path.join(root, 'adv')):
            os.makedirs(os.path.join(root, 'adv'))
        if not os.path.exists(os.path.join(root, 'show')):
            os.makedirs(os.path.join(root, 'show'))

        if not os.path.exists(os.path.join(root, 'adv_img')):
            os.makedirs(os.path.join(root, 'adv_img'))
        # if not os.path.exists(os.path.join(root, 'clip_img')):
        #     os.makedirs(os.path.join(root, 'clip_img'))

        hill_imgs = pre_hill.view(pre_hill.size(0), 1, im_size, im_size).repeat(1, 3, 1, 1)

        if modi_num >= 0.:
            for i in range(input_A.size(0)):
                clip_img = ToPILImage()((adv[i].data.cpu()))
                real_img = ToPILImage()((real_A[i].data.cpu()))
                adv_path = os.path.join(root, 'adv', image_names[i] + '_' + str(int(modi_num)) + '.png')
                clip_img.save(adv_path)
                real_path = os.path.join(root, 'clean', image_names[i] + '_' + str(int(modi_num)) + '.png')
                real_img.save(real_path)

                if True:
                    hill_img = ToPILImage()(hill_imgs[i].data.cpu())
                    temp_adv = torch.abs(adv_noise[i].data.cpu())
                    temp_adv = temp_adv / torch.max(temp_adv)
                    temp_adv = 1 - temp_adv
                    adv_img = ToPILImage()(temp_adv)

                    temp_hill = image_hill[i].data.cpu()

                    temp_hill = 1 - temp_hill
                    temp_hill = temp_hill.view(1, im_size, im_size).repeat(3, 1, 1)

                    temp_hill = ToPILImage()(temp_hill)
                    final = Image.fromarray(np.concatenate([temp_hill, hill_img, real_img, adv_img, clip_img], 1))
                    # final1 = temp_hill
                    # final2 = hill_img
                    # final3 = real_img
                    final4 = adv_img
                    # final5 = clip_img
                    final.save(os.path.join(root, 'show', image_names[i] + '_' + str(int(modi_num)) + '.png'))
                    # final1.save(os.path.join(root, 'temp_hill', image_names[i] + '_' + str(int(modi_num)) + '.png'))
                    # final2.save(os.path.join(root, 'hill_img', image_names[i] + '_' + str(int(modi_num)) + '.png'))
                    # final3.save(os.path.join(root, 'real_img', image_names[i] + '_' + str(int(modi_num)) + '.png'))
                    final4.save(os.path.join(root, 'adv_img', image_names[i] + '_' + str(int(modi_num)) + '.png'))
                    # final5.save(os.path.join(root, 'clip_img', image_names[i] + '_' + str(int(modi_num)) + '.png'))

def clip(adv_A, real_A, eps):
    g_x =  adv_A-real_A
    clip_gx = torch.clamp(g_x, min=-eps, max=eps)
    adv_x = real_A + clip_gx
    return adv_x

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()