# G-FRadv
## Towards Greedy Iterative Adversarial Attack with Distortion Maps against Deep Face Recognition
##  Environment

```
torch
torchvision
tensorboard
scikit-learn
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```
##  Download model
```
https://pan.baidu.com/s/1ElJlfmMwOGX699MsgLY8qA
```
##  Dowload dataset
```
https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw
```
##  Attack

```
Python FGR_attack --dataroot 1lingyugekaishi
```
##  Details
The data results are stored in the `result` file, where `L112` is the source file and `fil_112` is the target file (which is the file closest to the source file). The distorted images were generated using the DRMSFFN model.
```
https://github.com/liufeiqiang123/DRMSFFN
```
## Citation

```bibtex
@article{gao2025towards,
  title={Towards Greedy Iterative Adversarial Attack With Distortion Maps Against Deep Face Recognition},
  author={Gao, Peng and Zhu, Jiu-Ao and Qin, Wen-Hua},
  journal={IEEE Signal Processing Letters},
  volume={32},
  pages={4369--4373},
  year={2025},
  publisher={IEEE}
}
```
