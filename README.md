# batteryNET

Pipeline for deep-learning based image segmentation of volumetric microCT scans of a lithium-metal battery on NERSC.

## Description

Code to organize, prepare, and segment multi-scan, larger-than-memory, volumetric dataset. Deep learning segmentation uses pytorch-lightning for code organization, logging and metrics, and multi-GPU parallelization of training and prediction. Also uses moani for volumetric data augmentation and model implementation (currently U-Net++). Training can be done slice-by-slice with 2D models trained on 2D patches either aligned with or perpendicular to the electrode plane, or with 3D models on 3D patches. Sparse labels can be used to minimize required hand-labeling -- less than 0.1% of the entire dataset was labeled. Also includes code for data preparation (alignment/cropping/patching).

The code also has functionality for estimating segmentation uncertainty, both in the model parameters (systemic) and in the input data (aleatoric). I haven't done as much testing on this feature.

This pipeline was developed to segment a dataset consisting of 25 different gigapixel microCT scans of a lithium-metal battery collected over a single charge/discharge cycle lasting 4 hours. However code can be easily adapted for use general volumetric data.

## Getting Started

### Dependencies

* This code is written to be used on NERSC Perlmutter, though nothing wrong with trying on a powerful local workstation.

### Installing

* Download repo
```
git clone git@github.com:perlmutter/batteryNET.git
```
* Create environment
```
conda env create -f environment.yml
```
* If adapting to a *new dataset*, adjust filepaths in 2 files:
  *  [`constants.py`](https://github.com/perlmutter/batteryNET/blob/main/constants.py), such as path for saving results and 
  *  [`xsection-unet2d.json`](https://github.com/perlmutter/batteryNET/blob/main/setup_files/xsection-unet2d.json) setup file following example in folder [`setup_files`](https://github.com/perlmutter/batteryNET/tree/main/setup_files)

### Executing program

* On NERSC Perlmutter interactive node

  * create the model using settings specified in setup_files/xsection-unet2d.json:
  ```
  conda activate batteryNET
  python train.py setup_files/xsection-unet2d.json --num_workers 32 --epochs 200
  ```
  * test model on data:
  ```
  python predict_2d.py setup_files/xsection-unet2d.json
  ```

* Submitting NERSC batch job
  * create the model using settings specified in setup_files/xsection-unet2d.json:
  ```
  sbatch batch_scripts/train_unet2D.sh
  ```
  * test model on data:
  ```
  sbatch batch_scripts/predict_unet2d.sh
  ```

### Where is the model and segmentation results saved?
* For example, suppose your code is in /home/yourcode then a folder /home/battery_data will be created 

## Authors

David Perlmutter (dperl@lbl.gov), 
Dani Ushizima (dushizima@lbl.gov)

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

* Core segmentation code was based on this monai tutorial:
  * [Spleen 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
* Uncertainty estimation model was based on:
  * [Kendall, A and Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision](https://arxiv.org/abs/1703.04977)
