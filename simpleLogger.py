import collections
import os

import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


import pdb

class mySimpleLogger(LightningLoggerBase):
    def __init__(self,log_dir='./',keys=None):
        super().__init__()
        self.log_dir = log_dir
        self.keys = ['epoch','train_loss_step','train_loss_epoch','val_loss']
        for k in keys: self.keys.append(k)
        self.history = collections.defaultdict(list) # copy not necessary here
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "my_simple_logger"

    @property
    def experiment(self):
        # Return the experiment version, int or str.
        return "default"

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for metric_name, metric_value in metrics.items():
            if not metric_name in self.keys:
                continue

            if metric_name != 'epoch':
                self.history[metric_name].append(metric_value)
            else: # case epoch. We want to avoid adding multiple times the same. It happens for multiple losses.
                if (not len(self.history['epoch']) or    # len == 0:
                    not self.history['epoch'][-1] == metric_value) : # the last values of epochs is not the one we are currently trying to add.
                    self.history['epoch'].append(metric_value)
                else:
                    pass
        return

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        plt.figure()
        plt.plot(self.history["train_loss_epoch"],label='train')
        plt.plot(self.history["val_loss"],label='val')
        plt.xlabel('batch steps')
        plt.title("loss")
        plt.gca().legend()
        plt.tight_layout()
        filename = "loss.png"
        plt.savefig(os.path.join(self.log_dir,filename))
        plt.close()

        for key in self.history.keys():
            if not any([x in key for x in ["train_loss_epoch","val_loss","epoch"]]):
                plt.figure()
                plt.plot(self.history[key])
                plt.title(key)
                plt.tight_layout()
                filename = os.path.join(self.log_dir,f"{key}.png")
                plt.savefig(filename)
                plt.close()

        for key in self.history.keys():
            arr = np.asarray(self.history[key])
            if arr.ndim == 0:
                arr =  np.expand_dims(arr,axis=0)
            filename = os.path.join(self.log_dir,f"{key}.csv")
            np.savetxt(filename, arr, delimiter=',')
