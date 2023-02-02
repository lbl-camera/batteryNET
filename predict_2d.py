#!/usr/bin/env python
import os
import glob
import json
import re
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, util, color
import pytorch_lightning as pl
import torch
import torchmetrics
import matplotlib.pyplot as plt
from BatterySeg import BatterySeg
import constants as const
from dataset import StackDataset2D
from utils import overlap_predictions, postprocess_slice


import pdb


def main():
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument("config_file", type=str, help="json file containing data and model training parameters")
    parser.add_argument("--run_name", type=str, default='default', help="folder name to save all training results")
    parser.add_argument("--predict_stack", type=str, default="IM_298-1_240", help="stack to run prediction on")
    parser.add_argument("--predict_slices", type=list, default=None, help="slices to run prediction on")
    parser.add_argument("--epistemic_est", type=bool, default=False, help="whether to estimate epistemic uncertainty with MC dropout")
    parser.add_argument("--epistemic_iters", type=int, default=10, help="number of MC dropout iterations")
    # add DATA level args
    parser = StackDataset2D.add_args(parser)
    args = parser.parse_args()

    # get training vars from JSON file
    predict_args = _parse_prediction_variables(args)

    # load model
    weights_path = os.path.join(predict_args['weights_dir'],"model_weights.ckpt")
    model = BatterySeg.load_from_checkpoint(weights_path,**predict_args)

    # start from previous stoping point
    predict_dir = os.path.join(predict_args['prediction_save_dir'],predict_args['predict_stack'],const.SUBFOLDER_PRED)
    data_idx = [int(re.findall("img_([0-9]+).tif",x)[0]) for x in sorted(glob.glob(os.path.join(predict_dir,f"img_*.tif")))]
    if data_idx and data_idx[-1] > predict_args["predict_slices"][0]:
        predict_args["predict_slices"][0] = data_idx[-1]

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        default_root_dir=predict_args['prediction_save_dir'],
        gpus=predict_args['gpus'],
        strategy='dp'
        )
    print(f"Predicting {predict_args['predict_stack']}...")
    # create stack dataset/dataloader
    stack_dataset = StackDataset2D(predict_args["predict_stack"],predict_args["predict_slices"],predict_args)
    stack_loader = torch.utils.data.DataLoader(
                stack_dataset, batch_size=1, shuffle=False,
                num_workers=model.hparams.num_workers, pin_memory=True)
    # predict stack
    output = trainer.predict(model, dataloaders=stack_loader)

    # save predictions
    print(f"Saving {predict_args['predict_stack']} predictions to {predict_args['pred_dir']}...")
    for filename, image, probs, sigma2 in tqdm(output):
        image = image.numpy().squeeze()
        pred = torch.argmax(probs,dim=0).byte().squeeze().numpy()
        # postprocessing: only keep predictions in allowed slices defined in constants
        pred = postprocess_slice(pred,predict_args['predict_stack'])
        entropy = -torch.sum(probs * torch.log2(probs), axis=1)
        entropy = (entropy*(255/torch.log2(torch.tensor(model.hparams.num_classes)))).byte().squeeze().numpy()


        io.imsave(os.path.join(predict_args['pred_dir'],filename[0]), pred,check_contrast=False)
        io.imsave(os.path.join(predict_args['entropy_dir'],filename[0]), entropy)
        overlap = overlap_predictions(image, pred)
        io.imsave(os.path.join(predict_args['overlap_dir'],filename[0]), overlap)
        if model.hparams.aleatoric_est:
            sigma2 = (sigma2*(255.0/sigma2.max())).byte().squeeze().numpy()
            io.imsave(os.path.join(predict_args['sigma2_dir'],filename[0]), sigma2)
        else:
            if os.path.exists(predict_args['sigma2_dir']): os.rmdir(predict_args['sigma2_dir'])

def _parse_prediction_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    predict_args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(predict_args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        predict_args.update(config_dict)

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
