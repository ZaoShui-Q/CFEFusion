# CFEFusion

English | [简体中文](README.zh-CN.md)

CFEFusion is a PyTorch model for multimodal medical image fusion. It uses separate feature paths for the functional and anatomical inputs, then exchanges complementary information through local statistics alignment and cross-modal attention.

## Setup

Run the commands from the repository root. The following environment uses Python 3.10 and the CUDA 11.8 builds of PyTorch and torchvision.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install timm==1.0.22 numpy==2.1.2 opencv-python==4.12.0.88 Pillow==11.3.0
```

## Dataset Preparation and Augmentation

Before training, run the augmentation script on the SPECT–MRI training split. It extracts overlapping 160×160 patches with a stride of 10, using the same crop coordinates for each image pair to preserve spatial alignment.

```bash
python dataset/zengqiang.py --spect_dir data/SPECT-MRI/train/SPECT --mri_dir data/SPECT-MRI/train/MRI --output_dir data/SPECT-MRI/patches --crop_size 160 --stride 10
```

The script saves the paired patches in `data/SPECT-MRI/patches/SPECT` and `data/SPECT-MRI/patches/MRI`. Use these directories as `data_dir_vis` and `data_dir_ir` in `train.py`.

## Training

Set `data_dir_vis` and `data_dir_ir` to the paired training directories, `modelpth` to the checkpoint output directory, and `Method` to a run name. Run `python train.py` for SPECT–MRI or `python train_CT.py` for a separate grayscale CT–MRI training run. Training requires CUDA and saves a `fusion_model.pth` state dict in the run directory.

## Inference

Provide two directories of registered images. Paired images must have identical filenames and dimensions, with height and width divisible by 8.

In the inference script, set `fusion_model_path` to a compatible `Fusion(output=1)` checkpoint, `vi_path` to the SPECT, PET, or CT directory, `ir_path` to the MRI directory, and `fused_dir` to the output directory.

| Input pair | Command | Output |
| --- | --- | --- |
| SPECT–MRI or PET–MRI | `python test.py --gpu 0` | Color fusion preserving the SPECT or PET chrominance |
| CT–MRI | `python test_CT.py --gpu 0` | Grayscale fusion |

For cross-modal generalization, use the same SPECT–MRI checkpoint for PET–MRI and CT–MRI.
