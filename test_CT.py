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
            f"CT and MRI image counts differ. {len(self.filepath_vis)} vs {len(self.filepath_ir)}"

        assert self.filenames_vis == self.filenames_ir, \
            "Check that every CT image has an MRI image with the same filename"

        if length > 0:
            self.length = min(length, len(self.filepath_vis), len(self.filepath_ir))
        else:
            self.length = len(self.filepath_vis)

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        # Read both inputs as grayscale to bypass the pseudocolor YUV pipeline.
        image_vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        image_inf = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if image_vis is None:
            raise FileNotFoundError(f"Could not read the CT image at {vis_path}")
        if image_inf is None:
            raise FileNotFoundError(f"Could not read the MRI image at {ir_path}")

        if image_vis.shape[:2] != image_inf.shape[:2]:
            raise ValueError(
                f"Image sizes differ. {os.path.basename(vis_path)} {image_vis.shape[:2]} vs "
                f"{os.path.basename(ir_path)} {image_inf.shape[:2]}"
            )

        image_vis = image_vis.astype(np.float32) / 255.0
        image_ir = image_inf.astype(np.float32) / 255.0

        image_vis = np.expand_dims(image_vis, axis=0)
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

    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location=device))
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

            logits = fusionmodel(images_vis, images_ir)

            fusion_image = torch.clamp(logits, 0.0, 1.0)
            fused_image = fusion_image.squeeze(1).cpu().numpy()
            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :]
                image = Image.fromarray(image, mode='L')
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Successfully!'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CT-MRI Fusion with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    main()
