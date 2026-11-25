#!/usr/bin/python
# -*- encoding: utf-8 -*-
import argparse
import datetime
import time
import logging
import os
import warnings

import cv2
import numpy as np

from logger import setup_logger
from Fusionnet import Fusion
from loss import Fusionloss
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

warnings.filterwarnings('ignore')


def prepare_data_path(dataset_path):
    if dataset_path is None:
        raise ValueError("Set the dataset directory before training")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist at {dataset_path}")

    image_exts = (".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".png")
    data = []
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(image_exts):
            data.append(filepath)

    data.sort()
    filenames = [os.path.basename(path) for path in data]
    return data, filenames


class Fusion_dataset(Dataset):
    def __init__(self):
        super(Fusion_dataset, self).__init__()

        # CT feeds the functional path and MRI feeds the structural path.
        data_dir_vis = r"....."
        data_dir_ir = r"....."

        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)

        if len(self.filepath_vis) == 0 or len(self.filepath_ir) == 0:
            raise RuntimeError(
                f"Check the training directories. No images were found in at least one modality. CT={len(self.filepath_vis)}, MRI={len(self.filepath_ir)}"
            )

        assert len(self.filepath_vis) == len(self.filepath_ir), \
            f"CT and MRI image counts differ. {len(self.filepath_vis)} vs {len(self.filepath_ir)}"

        assert self.filenames_vis == self.filenames_ir, \
            "Check that every CT image has an MRI image with the same filename"

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

        image_vis = np.asarray(image_vis, dtype=np.float32) / 255.0
        image_ir = np.asarray(image_inf, dtype=np.float32) / 255.0

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


def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def train_fusion(num=0, logger=None):
    if logger is None:
        logger = logging.getLogger()

    lr_start = 0.0002
    modelpth = '.....'
    Method = '...'
    modelpth = os.path.join(modelpth, Method)

    fusionmodel = eval('Fusion')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)

    train_dataset = Fusion_dataset()
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = Fusionloss().cuda()

    epoch = 2
    st = glob_st = time.time()
    logger.info('Training CT-MRI Fusion Model start~')

    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo

        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = image_vis.cuda(non_blocking=True)
            image_ir = image_ir.cuda(non_blocking=True)

            if image_vis.size(1) != 1:
                image_vis = image_vis[:, 0:1, :, :]
            if image_ir.size(1) != 1:
                image_ir = image_ir[:, 0:1, :, :]

            logits = fusionmodel(image_vis, image_ir)

            optimizer.zero_grad()

            loss_fusion, loss_in, ssim_loss, loss_grad = criteria_fusion(
                image_vis=image_vis,
                image_ir=image_ir,
                fused=logits
            )
            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))

            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'ssim_loss: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=ssim_loss.item(),
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed

    os.makedirs(modelpth, exist_ok=True)
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("CT-MRI Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CT-MRI with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    for i in range(1):
        train_fusion(i, logger)
        print("|{0} Train CT-MRI Fusion Model Successfully~!".format(i + 1))
    print("training Done!")
