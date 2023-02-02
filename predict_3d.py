#!/usr/bin/env python
import os
import glob
import json
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, util, color
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt

import constants as const
from BatterySeg import BatterySeg
from dataset import StackDataset3D
from utils import overlap_predictions, postprocess_slice

import pdb


def main():
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument("config_file", type=str, help="json file containing data and model training parameters")
    parser.add_argument("--run_name", type=str, default='default', help="folder name to save all training results")
    parser.add_argument("--predict_stack", type=str, default="IM_298-1_240", help="stacks to run prediction on")
    parser.add_argument("--predict_slices", type=list, default=[210, 311], help="slices to run prediction on")
    parser.add_argument("--epistemic_est", type=bool, default=False, help="whether to estimate epistemic uncertainty with MC dropout")
    parser.add_argument("--epistemic_iters", type=int, default=10, help="number of MC dropout iterations")
    # add DATA level args
    parser = StackDataset3D.add_args(parser)
    args = parser.parse_args()

    # get training vars from JSON file
    predict_args = _parse_prediction_variables(args)

    # load model
    weights_path = os.path.join(predict_args['weights_dir'],"model_weights.ckpt")
    model = BatterySeg.load_from_checkpoint(weights_path,**predict_args)

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        default_root_dir=predict_args['prediction_save_dir'],
        gpus=predict_args['gpus'],
        strategy='dp'
        )
    print(f"Predicting {predict_args['predict_stack']}...")
    # create stack dataset/dataloader
    stack_dataset = StackDataset3D(predict_args["predict_stack"],predict_args["predict_slices"],predict_args)
    stack_loader = torch.utils.data.DataLoader(
                stack_dataset, batch_size=1, shuffle=False,
                num_workers=model.hparams.num_workers, pin_memory=True)
    # predict stack
    output = trainer.predict(model, dataloaders=stack_loader)
    (idx, vol, probability, sigma2) = zip(*output)
    del output, vol, sigma2, model, trainer, stack_loader
    # reassemble volume from chunks
    print(f"Aggregating output patches...")
    pdb.set_trace()
    # vol = stack_dataset.aggregate_patches(vol,idx).squeeze() # remove channel dim
    probability = stack_dataset.aggregate_patches(probability,idx).transpose(2,0) # swap channel and xsectional dim (want to iterate over xsectional dim)
    if model.hparams.aleatoric_est:
        sigma2 = stack_dataset.aggregate_patches(sigma2,idx).squeeze() # remove channel dim
        sigma2 = (sigma2*(255.0/sigma2.max())).byte() # normalize and scale to 0-to-255
    # save predictions
    vol = h5py.File(self.data_file,'r')
    print(f"Saving {predict_args['predict_stack']} predictions to {predict_args['pred_dir']}...")
    for i in range(probability.shape[0]):
        filename = f"img_{i:04d}{const.EXT}"
        pred = torch.argmax(probs[i],dim=0).byte().squeeze().numpy()
        pred = postprocess_slice(pred,predict_args['predict_stack'])
        entropy = -torch.sum(probs * torch.log2(probs), axis=0)
        entropy = (entropy*(255/torch.log2(torch.tensor(model.hparams.num_classes)))).byte().squeeze().numpy()

        io.imsave(os.path.join(predict_args['pred_dir'],filename),pred,check_contrast=False)
        io.imsave(os.path.join(predict_args['entropy_dir'],filename),entropy)
        image = vol['data'][:,i,:]
        overlap = overlap_predictions(image, pred)
        io.imsave(os.path.join(predict_args['overlap_dir'],filename),overlap)
        if model.hparams.aleatoric_est:
            sigma2[i] = sigma2[i].numpy()
            if os.path.exists(predict_args['sigma2_dir']): os.rmdir(predict_args['sigma2_dir'])

def _parse_prediction_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    predict_args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(predict_args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        predict_args.update(config_dict)

    predict_args['predict_slices'] = range(predict_args['predict_slices'][0],predict_args['predict_slices'][-1])
    # predict_args['gpus'] = -1 if torch.cuda.is_available() else 0
    predict_args['gpus'] = 1 if torch.cuda.is_available() else 0

    predict_args['weights_dir'] = os.path.join(const.SAVE_BASE,const.SUBFOLDER_RESULTS,predict_args['run_name'])
    predict_args['prediction_save_dir'] = os.path.join(const.SAVE_BASE,const.SUBFOLDER_RESULTS,predict_args['run_name'],const.SUBFOLDER_DATA_PRED)

    predict_args['pred_dir'] = os.path.join(predict_args['prediction_save_dir'],predict_args['predict_stack'],const.SUBFOLDER_PRED)
    predict_args['entropy_dir'] = os.path.join(predict_args['prediction_save_dir'],predict_args['predict_stack'],const.SUBFOLDER_ENTROPY)
    predict_args['overlap_dir'] = os.path.join(predict_args['prediction_save_dir'],predict_args['predict_stack'],const.SUBFOLDER_OVER)
    predict_args['sigma2_dir'] = os.path.join(predict_args['prediction_save_dir'],predict_args['predict_stack'],const.SUBFOLDER_ALEATORIC)
    folders = [predict_args['pred_dir'], predict_args['entropy_dir'], predict_args['overlap_dir'], predict_args['sigma2_dir']]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    return predict_args

if __name__ == "__main__":
    main()
