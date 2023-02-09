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
git clone https://github.com/lbl-camera/batteryNET.git 
```
* Create environment
```
conda env create -f environment.yml
```
* If adapting to a *new dataset*, adjust filepaths in 2 files:
  *  [`constants.py`](https://github.com/lbl-camera/batteryNET/blob/main/constants.py), such as path for saving results and 
  *  [`xsection-unet2d.json`](https://github.com/lbl-camera/batteryNET/blob/main/setup_files/xsection-unet2d.json) setup file following example in folder [`setup_files`](https://github.com/lbl-camera/batteryNET/tree/main/setup_files)

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

## Copyright Notice

batteryNET Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


## License Agreement

batteryNET Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works thereof,
in binary and source code form.


## References

* Core segmentation code was based on this monai tutorial:
  * [Spleen 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d_lightning.ipynb)
* Uncertainty estimation model was based on:
  * [Kendall, A and Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision](https://arxiv.org/abs/1703.04977)
