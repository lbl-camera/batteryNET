#!/usr/bin/env python

import os
import json
import csv
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from monai.networks.nets import UNet

import constants as const
from BatterySeg import BatterySeg
from dataset import PreCroppedDataset, CropDataset3D
from simpleLogger import mySimpleLogger

import pdb


def main():
    parser = ArgumentParser(conflict_handler='resolve')
    # add PROGRAM level args
    parser.add_argument("config_file", type=str, help="json file containing data and model training parameters")
    parser.add_argument("--run_name", type=str, default='default', help="folder name to save all training results")
    parser.add_argument("--start_from_checkpoint", type=bool, default=False, help="whether to start from new model or previous checkpoint")
    parser.add_argument("--gpus", type=int, default=None, help="how many gpus to use")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train for")
    parser.add_argument("--testing_data_dir", type=str, default=None, help="path to test data, if any")
    # add MODEL and DATA level args
    parser = BatterySeg.add_args(parser)
    parser = PreCroppedDataset.add_args(parser)
    parser = CropDataset3D.add_args(parser)
    args = parser.parse_args()

    # get training vars from JSON file
    train_args = _parse_training_variables(args)

    # initialise the LightningModule
    if train_args['start_from_checkpoint']:
            weights_path = os.path.join(train_args['save_dir'],"model_weights.ckpt")
            model = BatterySeg.load_from_checkpoint(weights_path,**train_args)
    else:
        model = BatterySeg(**train_args)

    # set up loggers and checkpoints
    my_logger = mySimpleLogger(log_dir=train_args['training_save_dir'],
                                keys=['val_acc','val_prec','val_recall','val_iou'])
    tb_logger = pl.loggers.TensorBoardLogger(
                                save_dir=train_args['training_save_dir'],
                                name=None,
                                version=None)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                 dirpath=train_args['training_save_dir'],
                                 filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
                                 save_top_k=1,
                                 # every_n_train_steps=100,
                                 every_n_epochs=1,
                                 save_weights_only=True,
                                 verbose=True,
                                 monitor="val_acc",
                                 mode='max')
    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss',
                                min_delta=1e-3,
                                patience=10,
                                verbose=True,
                                mode='min')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=False)

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        default_root_dir=train_args['training_save_dir'],
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=100,
        enable_checkpointing=True,
        logger=[tb_logger, my_logger],
        gpus=train_args['gpus'],
        # gpus=1, # for debugging
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        max_epochs=train_args['epochs'],
        max_time='00:00:50:00'
        )
    # train
    trainer.fit(model)
    # save best model
    print("Done training...saving models")
    model = BatterySeg.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.checkpoint_callback.save_weights_only = False
    trainer.save_checkpoint(os.path.join(train_args['save_dir'],"model_weights.ckpt"))
    dummy_input = torch.empty((1, 1, *train_args['train_patch_size']), dtype=torch.float)
    torch.onnx.export(model,dummy_input,
                        os.path.join(train_args['save_dir'],"model.onnx"),
                        export_params=True,
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                      'output' : {0 : 'batch_size'}}
                        )

    if train_args['testing_data_dir']:
        train_args['valid_split'] = 1.0
        dummy_ds = PreCroppedDataset(train_args,validation=False)
        test_ds = PreCroppedDataset(train_args,validation=True, idx=dummy_ds.get_valid_idx())
        test_loader = torch.utils.data.DataLoader(
                    test_ds, batch_size=1, shuffle=False,
                    num_workers=train_args['num_workers'], pin_memory=True)
        test_results = trainer.test(model,dataloaders=test_loader)
        test_results = test_results[0] # unpack list

        with open(os.path.join(train_args['training_save_dir'],'test_results.csv'), 'w') as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, test_results.keys())
            w.writeheader()
            w.writerow(test_results)

def _parse_training_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    train_args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(train_args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        train_args.update(config_dict)

    train_args['train_patch_size'] = tuple(train_args['train_patch_size']) # tuple expected, not list
    if train_args['gpus'] is None:
        train_args['gpus'] = -1 if torch.cuda.is_available() else 0

    train_args['save_dir'] = os.path.join(const.SAVE_BASE,const.SUBFOLDER_RESULTS,train_args['run_name'])
    train_args['training_save_dir'] = os.path.join(const.SAVE_BASE,const.SUBFOLDER_RESULTS,train_args['run_name'],const.SUBFOLDER_TRAINING)

    return train_args

if __name__ == "__main__":
    main()
