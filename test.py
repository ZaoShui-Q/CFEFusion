# coding:utf-8
import os
import argparse
import numpy as np
import glob
import cv2

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from Fusionnet import Fusion
from PIL import Image


def prepare_data_path(dataset_path):
    data = glob.glob(os.path.join(dataset_path, "*.bmp"))
    data.extend(glob.glob(os.path.join(dataset_path, "*.tif")))
    data.extend(glob.glob(os.path.join(dataset_path, "*.jpg")))
    data.extend(glob.glob(os.path.join(dataset_path, "*.png")))
    data.sort()

    filenames = [os.path.basename(path) for path in data]

    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, length=0):
        super(Fusion_dataset, self).__init__()
        assert split == 'test', 'split must be "test"'

        data_dir_vis = vi_path
        data_dir_ir = ir_path

        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
        self.split = split

        assert len(self.filepath_vis) == len(self.filepath_ir), \
            f"Functional and structural image counts differ. {len(self.filepath_vis)} vs {len(self.filepath_ir)}"

        assert self.filenames_vis == self.filenames_ir, \
            "Check that every functional image has a structural image with the same filename"
        
        if length > 0:
            self.length = min(length, len(self.filepath_vis), len(self.filepath_ir))
        else:
            self.length = len(self.filepath_vis)

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        image_vis = cv2.imread(vis_path)
        # Convert OpenCV's BGR output to the RGB order expected by RGB2YUV.
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)

        image_inf = cv2.imread(ir_path, 0)

        image_vis = image_vis.astype(np.float32).transpose((2, 0, 1)) / 255.0
        image_ir = image_inf.astype(np.float32) / 255.0
        image_ir = np.expand_dims(image_ir, axis=0)

        name = self.filenames_vis[index]

        return (
            torch.tensor(image_vis),
            torch.tensor(image_ir),
            name,
        )

    def __len__(self):
        return self.length

# Fill in the checkpoint path, paired input directories, and output directory.
def main():
    fusion_model_path = r'.....'     

    ir_path = r'.....'                     
    vi_path = r'.....'                      
    fused_dir = r'.....'             

    os.makedirs(fused_dir, mode=0o777, exist_ok=True)

    fusionmodel = Fusion(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.gpu >= 0:
        fusionmodel.to(device)

    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel.eval()
    print('fusionmodel load done!')

    test_dataset = Fusion_dataset('test', ir_path=ir_path, vi_path=vi_path) 

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):

            if args.gpu >= 0:
                images_vis = images_vis.to(device)
                images_ir = images_ir.to(device)

            images_vis_yuv = RGB2YUV(images_vis)
            images_vis_y = images_vis_yuv[:, 0:1, :, :]
            logits = fusionmodel(images_vis_y, images_ir)

            fusion_yuv = torch.cat(
                (logits, images_vis_yuv[:, 1:2, :, :], images_vis_yuv[:, 2:, :, :]),
                dim=1,
            )

            fusion_image = YUV2RGB(fusion_yuv)
            fusion_image = torch.clamp(fusion_image, 0.0, 1.0)

            fused_image = fusion_image.detach().cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))

            # Preserve the absolute [0, 1] intensity scale when converting to uint8.
            fused_image = np.clip(fused_image, 0.0, 1.0)
            fused_image = np.uint8(np.round(fused_image * 255.0))

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Successfully!'.format(save_path))


def YUV2RGB(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)

    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)

    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)

    temp = (im_flat + bias).mm(mat).to(device)

    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )

    return out


def RGB2YUV(input_im):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # [N, C, H, W] -> [N * H * W, C]

    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = (R - Y) * 0.713 + 0.5
    V = (B - Y) * 0.564 + 0.5

    Y = torch.unsqueeze(Y, 1)
    U = torch.unsqueeze(U, 1)
    V = torch.unsqueeze(V, 1)

    temp = torch.cat((Y, U, V), dim=1).to(device)

    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Fusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()