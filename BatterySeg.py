#!/usr/bin/env python
import constants as const

import pytorch_lightning as pl
import torch
import torchmetrics
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import list_data_collate, decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from dataset import PreCroppedDataset, RandomCropDataset3D, GridCropDataset3D
from monai.config import print_config
from monai.apps import download_and_extract
import matplotlib.pyplot as plt
import os
import glob
import sys
from argparse import ArgumentParser

import pdb



class BatterySeg(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        # self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        if self.hparams.aleatoric_est:
            self.hparams.out_channels =  2*self.hparams.num_classes
            self.class_weights = torch.ones(self.hparams.out_channels) # set all weights to one
            self.class_weights[self.hparams.background_index] = self.hparams.background_weight # chagne background weight
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights,ignore_index=self.hparams.unlabeled_index, reduction='none')
            # self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.unlabeled_index, reduction='none')
        else:
            self.hparams.out_channels = self.hparams.num_classes
            self.class_weights = torch.ones(self.hparams.out_channels) # set all weights to one
            self.class_weights[self.hparams.background_index] = self.hparams.background_weight # chagne background weight
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights,ignore_index=self.hparams.unlabeled_index, reduction='mean')
            # self.cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.unlabeled_index, reduction='mean')

        self.cnfmat = torchmetrics.ConfusionMatrix(num_classes=self.hparams.num_classes,
                                                    normalize=None)

        if len(self.hparams.train_patch_size) != self.hparams.spatial_dims:
            print("dimension of train_patch_size does not equal spatial_dims")
            sys.exit()

        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=1,
            out_channels=self.hparams.out_channels,
            channels=(32, 64, 128, 256, 512, 512, 512),
            strides=(2, 2, 2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            dropout=0.2,
            norm=Norm.BATCH,
        )

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("model")
        parser.add_argument("--spatial_dims", type=int, default=2, help="2D or 3D data")
        parser.add_argument("--num_classes", type=int, default=4, help="number of segmentation classes to predict")
        parser.add_argument("--unlabeled_index", type=int, default=-1, help="unlabeled data index")
        parser.add_argument("--background_index", type=int, default=0, help="background index")
        parser.add_argument("--background_weight", type=int, default=1, help="relative weight of background class (<=1, 1 is same weight as foreground)")
        parser.add_argument("--aleatoric_est", type=str, default=False, help="whether to estimate aleatoric uncertainty in training")
        parser.add_argument("--aleatoric_iters", type=int, default=30, help="number of MC dropout iterations")
        parser.add_argument("--batch_size", type=int, default=4, help="dataloader batch size")
        parser.add_argument("--steps_per_epoch", type=int, default=1000, help="number of steps to take per epoch (training + validation)")
        parser.add_argument("--num_workers", type=int, default=32, help="cpus used per dataloader")
        parser.add_argument("--lr", type=int, default=3e-4, help="Adam learning rate")
        parser.add_argument("--use_randcrop", type=bool, default=False, help="Use on-the-fly, random cropping of training data (3D volumes stored in hdf5 files). Default is to use pre-cropped tiff files")
        return parent_parser

    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        if stage == "fit":
            # set deterministic training for reproducibility
            set_determinism(seed=0)
            if self.hparams.use_randcrop:
                # train_ds = [ GridCropDataset3D(stack,self.hparams,validation=False)
                #            for stack in self.hparams.training_stacks ]
                # val_ds = [ GridCropDataset3D(stack, self.hparams,validation=True, idx=train_ds[i].get_valid_idx())
                #            for i, stack in enumerate(self.hparams.training_stacks) ]
                train_ds = [ RandomCropDataset3D(stack,self.hparams,validation=False)
                           for stack in self.hparams.training_stacks ]
                val_ds = [ RandomCropDataset3D(stack, self.hparams,validation=True)
                           for stack in self.hparams.training_stacks ]
                self.train_ds = torch.utils.data.ConcatDataset(train_ds)
                self.val_ds = torch.utils.data.ConcatDataset(val_ds)
            else:
                self.train_ds = PreCroppedDataset(self.hparams,validation=False)
                self.val_ds = PreCroppedDataset(self.hparams,validation=True, idx=self.train_ds.get_valid_idx())

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
            collate_fn=list_data_collate, num_workers=self.hparams.num_workers,
            persistent_workers=True, pin_memory=torch.cuda.is_available())
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=10,
                                                               min_lr=1e-6,
                                                               verbose=True)
        lr_scheduler = {"scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch"}

        return [optimizer], [lr_scheduler]

    def loss_function(self, logits, labels):
        if self.hparams.aleatoric_est:
            """ heteroscedastic classification network loss
                from "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" by Kendal and Gal
                https://arxiv.org/abs/1703.04977
            """
            logits, sigma2 = torch.tensor_split(logits,2,dim=1) # second half of logit channels are actually variance estimates
            torch.nn.functional.relu_(sigma2) # variance must be positive
            stochastic_probs = torch.stack([ self.cross_entropy_loss(
                                    logits + sigma2*torch.randn(sigma2.shape,device=sigma2.device),
                                    labels)
                                    for _ in range(self.hparams.aleatoric_iters)],dim=1) # first half of eq 12 in paper
            loss = torch.log(torch.exp(stochastic_probs).mean(dim=1)).mean() # second half of equation 12
        else:
            loss = self.cross_entropy_loss(logits, labels)

        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)
        if self.hparams.aleatoric_est:
            logits, _ = torch.tensor_split(logits,2,dim=1) # second half of logit channels are actually variance estimates
        # remove all unlabeled pixels from confusion matrix logging
        labeled_ind = (labels != self.hparams.unlabeled_index)
        labeled_logits = torch.cat([logits[i,:,labeled_ind[i,:,:]] for i in range(len(logits))],dim=-1).transpose(0,1)
        labeled_labels = torch.cat([labels[i,labeled_ind[i,:,:]] for i in range(len(labels))],dim=-1)
        self.cnfmat(labeled_logits, labeled_labels)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        acc, prec, recall, iou, _ = self._compute_cnf_stats()

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_prec', prec, prog_bar=False)
        self.log('val_recall', recall, prog_bar=False)
        self.log('val_iou', iou,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        logits = self.forward(images)
        loss = self.loss_function(logits, labels)
        if self.hparams.aleatoric_est:
            logits, _ = torch.tensor_split(logits,2,dim=1) # second half of logit channels are actually variance estimates
        labeled_ind = (labels != self.hparams.unlabeled_index)
        labeled_logits = torch.cat([logits[i,:,labeled_ind[i,:,:]] for i in range(len(logits))],dim=-1).transpose(0,1)
        self.cnfmat(labeled_logits, labels[labeled_ind])
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        acc, prec, recall, iou, iou_per_class = self._compute_cnf_stats()

        self.log('test_acc', acc, prog_bar=True, rank_zero_only=True)
        self.log('test_prec', prec, prog_bar=True, rank_zero_only=True)
        self.log('test_recall', recall, prog_bar=True, rank_zero_only=True)
        self.log('test_iou', iou,  prog_bar=True, rank_zero_only=True)
        for i, class_iou in enumerate(iou_per_class):
            self.log(f"test_iou-class_{i}", class_iou,  prog_bar=True, rank_zero_only=True)


    def pred_function(self, image):
        if self.hparams.spatial_dims == 2:
            # StackDataset2D class: input is entire slice, aggregate patches using sliding window inference
            return sliding_window_inference(image, self.hparams.pred_patch_size, 1, self.forward)
        elif self.hparams.spatial_dims == 3:
            # StackDataset3D class: input is chunk, chunks will be aggregated afterwards by StackDataset3D
            return self.forward(image)
        else:
            sys.exit("sptial_dims must be 2 or 3")

    def predict_step(self, batch, batch_idx):
        image, metadata = batch

        if self.hparams.epistemic_est:
            self.model.train() # enable Monte Carlo Dropout
            logits = torch.stack(
                    [ self.pred_function(image) for _ in range(self.hparams.epistemic_iters) ],
                    dim=1).mean(dim=1) # stack monte carlo iters, then average them
        else:
            logits = self.pred_function(image)
        if self.hparams.aleatoric_est:
            logits, sigma2 = torch.tensor_split(logits,2,dim=1) # second half of logit channels are actually variance estimates
            torch.nn.functional.relu_(sigma2)
            sigma2 = sigma2.sum(dim=1) # average variance across all channels
            sigma2 = sigma2.squeeze(dim=0) # scale, convert to ubyte, squeeze
        else:
            sigma2 = None

        probs = torch.nn.functional.softmax(logits,dim=1).squeeze(dim=0)
        image = (image*255).byte().squeeze(dim=0)
        return (metadata, image, probs, sigma2)

    def _compute_cnf_stats(self):

        import numpy as np

        cnfmat = self.cnfmat.compute()
        cnfmat = cnfmat.cpu().numpy()

        true = np.diag(cnfmat)
        tn = true[self.hparams.background_index]
        tp = np.delete(true, self.hparams.background_index)
        fn = np.delete(cnfmat.sum(1) - true, self.hparams.background_index)
        fp = np.delete(cnfmat.sum(0) - true, self.hparams.background_index)

        acc = np.nansum(true)/np.nansum(cnfmat)
        precision = np.nansum(tp)/np.nansum(tp + fp)
        recall = np.nansum(tp)/np.nansum(tp + fn)
        iou = np.nansum(tp)/(np.nansum(cnfmat) - tn)
        iou_per_class = tp / (tp + fp + fn)

        self.cnfmat.reset()

        return acc, precision, recall, iou, iou_per_class
