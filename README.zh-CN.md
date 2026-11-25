# CFEFusion

[English](README.md) | 简体中文

CFEFusion 是一个基于 PyTorch 的多模态医学图像融合模型。它分别提取功能图像和解剖图像的特征，再通过局部统计对齐和跨模态注意力交换互补信息。

## 环境安装

在仓库根目录执行以下命令。此示例使用 Python 3.10 和 CUDA 11.8 版本的 PyTorch 与 torchvision。

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install timm==1.0.22 numpy==2.1.2 opencv-python==4.12.0.88 Pillow==11.3.0
```

## 数据准备与增强

训练前，先对 SPECT–MRI 训练集做图像增强。脚本以 10 像素为步长提取重叠的 `160×160` 图像块，并对每对图像使用相同的裁剪坐标，以保持空间对齐。

```bash
python dataset/zengqiang.py --spect_dir data/SPECT-MRI/train/SPECT --mri_dir data/SPECT-MRI/train/MRI --output_dir data/SPECT-MRI/patches --crop_size 160 --stride 10
```

裁剪后的图像块分别保存在 `data/SPECT-MRI/patches/SPECT` 和 `data/SPECT-MRI/patches/MRI`。在 `train.py` 中，将这两个目录分别设为 `data_dir_vis` 和 `data_dir_ir`。

## 训练

将 `data_dir_vis` 和 `data_dir_ir` 设为对应的训练数据目录，`modelpth` 设为权重输出目录，`Method` 设为本次训练的名称。运行 `python train.py` 训练 SPECT–MRI 模型，或运行 `python train_CT.py` 单独训练灰度 CT–MRI 模型。训练需要 CUDA，模型的 `state_dict` 保存在本次训练目录下的 `fusion_model.pth` 中。

## 推理

将已配准的两种模态图像分别放在两个目录中。每对图像的文件名和尺寸必须相同，且图像高和宽均须为 8 的倍数。

在推理脚本中，将 `fusion_model_path` 设为与 `Fusion(output=1)` 兼容的权重文件，`vi_path` 设为 SPECT、PET 或 CT 图像目录，`ir_path` 设为 MRI 图像目录，`fused_dir` 设为融合图像的输出目录。

| 输入图像对             | 命令                          | 输出                                       |
| ---------------------- | ----------------------------- | ------------------------------------------ |
| SPECT–MRI 或 PET–MRI | `python test.py --gpu 0`    | 彩色融合图像，保留 SPECT 或 PET 的色度信息 |
| CT–MRI                | `python test_CT.py --gpu 0` | 灰度融合图像                               |

评估跨模态泛化时，PET–MRI 和 CT–MRI 均使用同一份在 SPECT–MRI 上训练的权重。
