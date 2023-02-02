import os
import subprocess
import numpy as np

import pdb

# pre/post-pocessing parameters
MIN_LI_VALUE = 40
MIN_INST_SIZE = 6

# defining folder structure.
# base folder
DATA_BASE =  '//global/cfs/cdirs/als/users/ivzenyuk/07292021-NewBattery-298_1/roi'
LABEL_BASE =  '//global/cfs/cdirs/als/users/ivzenyuk/07292021-NewBattery-298_1/labels'
STACKS = [
            'IM_298-1_000', 'IM_298-1_010', 'IM_298-1_020', 'IM_298-1_030', 'IM_298-1_040',
            'IM_298-1_050', 'IM_298-1_060', 'IM_298-1_070', 'IM_298-1_080', 'IM_298-1_090',
            'IM_298-1_100', 'IM_298-1_110', 'IM_298-1_120','IM_298-1_130', 'IM_298-1_140',
            'IM_298-1_150', 'IM_298-1_160', 'IM_298-1_170', 'IM_298-1_180', 'IM_298-1_190',
            'IM_298-1_200','IM_298-1_210', 'IM_298-1_220', 'IM_298-1_230', 'IM_298-1_240'
            ]
SAVE_BASE = os.path.join(subprocess.check_output('echo $SCRATCH',shell=True).decode("utf-8")[:-1],'battery_data')

# how many images are available for training (not for precropped dataset).
TRAIN_IMAGES_PER_EPOCH = 100
VAL_IMAGES_PER_EPOCH = 10

# the main subfolders, and extensions.
SUBFOLDER_IMAGE = 'image'
SUBFOLDER_CHUNK = 'chunk'
SUBFOLDER_LABEL = 'label'
SUBFOLDER_RESULTS = 'results'
SUBFOLDER_TRAINING = 'training'
SUBFOLDER_WEIGHT_IMGS = 'weight_images'
SUBFOLDER_DATA_PRED = 'prediction'
SUBFOLDER_PRED = 'label'
SUBFOLDER_OVER = 'overlap'
SUBFOLDER_ALEATORIC = 'aleatoric_var'
SUBFOLDER_ENTROPY = 'entropy'
SUBFOLDER_DATA_EVAL = 'evaluation'
SUBFOLDER_INSTANCE_SNIPPETS = 'instances'
SUBFOLDER_COMPARISON = 'comparison'
EXT = '.tif'

LABEL_COLORS = [
            (0,0,0), # label 0 (background) = transparent
            (1,0,0), # label 1 (dendrite) = red
            (0,0,1), # label 2 (pit) = blue
            (1,1,0), # label 3 (liquid film) = yellow
            (0,1,1), # label 4 (electrode) = green
            (1,0,1), # label 5 (deopsited lithium) = magenta
]

''' dictionary specifying in-plane slices in which each class is allowed '''
# initialize by allowing all 5 classes in all slices
all_slices_dict = {i: list(range(0,698)) for i in range(6)}
ALLOWED_SLICES = {stack: all_slices_dict.copy() for stack in STACKS}
# hardcode specific regions for each stack
# 000
for i in [1,2,3,5]: ALLOWED_SLICES["IM_298-1_000"][i] = None # no dendrites, pits, film, or deposited Li in pristine stack
ALLOWED_SLICES["IM_298-1_000"][4] = list(range(145,270)) + list(range(460,580)) # anode/cathode regions
# 010
ALLOWED_SLICES["IM_298-1_010"][1] = list(range(240,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_010"][2] = list(range(450,490)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_010"][3] = None # no film
ALLOWED_SLICES["IM_298-1_010"][4] = list(range(145,270)) + list(range(460,580)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_010"][5] = list(range(237,266)) # anode/electrolyte interface
# 020
ALLOWED_SLICES["IM_298-1_020"][1] = list(range(220,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_020"][2] = list(range(450,505)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_020"][3] = None # no film
ALLOWED_SLICES["IM_298-1_020"][4] = list(range(125,290)) + list(range(460,593)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_020"][5] = list(range(217,297)) # anode/electrolyte interface
# 030
ALLOWED_SLICES["IM_298-1_030"][1] = list(range(235,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_030"][2] = list(range(450,502)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_030"][3] = None # no film
ALLOWED_SLICES["IM_298-1_030"][4] = list(range(140,290)) + list(range(470,590)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_030"][5] = list(range(232,288)) # anode/electrolyte interface
# 040
ALLOWED_SLICES["IM_298-1_040"][1] = list(range(230,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_040"][2] = list(range(450,505)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_040"][3] = None # no film
ALLOWED_SLICES["IM_298-1_040"][4] = list(range(140,290)) + list(range(470,590)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_040"][5] = list(range(237,270)) # anode/electrolyte interface
# 050
ALLOWED_SLICES["IM_298-1_050"][1] = list(range(230,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_050"][2] = list(range(450,500)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_050"][3] = None # no film
ALLOWED_SLICES["IM_298-1_050"][4] = list(range(137,270)) + list(range(460,585)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_050"][5] = list(range(229,270)) # anode/electrolyte interface
# 060
ALLOWED_SLICES["IM_298-1_060"][1] = list(range(255,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_060"][2] = list(range(450,500)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_060"][3] = None # no film
ALLOWED_SLICES["IM_298-1_060"][4] = list(range(140,255)) + list(range(470,580)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_060"][5] = list(range(233,270)) # anode/electrolyte interface
# 070
ALLOWED_SLICES["IM_298-1_070"][1] = list(range(223,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_070"][2] = list(range(470,502)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_070"][3] = None # no film
ALLOWED_SLICES["IM_298-1_070"][4] = list(range(132,266)) + list(range(466,585)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_070"][5] = list(range(223,280)) # anode/electrolyte interface
# 080
ALLOWED_SLICES["IM_298-1_080"][1] = list(range(245,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_080"][2] = list(range(470,498)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_080"][3] = None # no film
ALLOWED_SLICES["IM_298-1_080"][4] = list(range(135,250)) + list(range(468,575)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_080"][5] = list(range(225,265)) # anode/electrolyte interface
# 090
ALLOWED_SLICES["IM_298-1_090"][1] = list(range(245,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_090"][2] = list(range(470,498)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_090"][3] = None # no film
ALLOWED_SLICES["IM_298-1_090"][4] = list(range(135,245)) + list(range(473,575)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_090"][5] = list(range(229,268)) # anode/electrolyte interface
# 100
ALLOWED_SLICES["IM_298-1_100"][1] = list(range(245,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_100"][2] = list(range(465,504)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_100"][3] = None # no film
ALLOWED_SLICES["IM_298-1_100"][4] = list(range(125,255)) + list(range(462,582)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_100"][5] = list(range(212,279)) # anode/electrolyte interface
# 110
ALLOWED_SLICES["IM_298-1_110"][1] = list(range(245,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_110"][2] = list(range(465,502)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_110"][3] = None # no film
ALLOWED_SLICES["IM_298-1_110"][4] = list(range(125,247)) + list(range(465,578)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_110"][5] = list(range(208,275)) # anode/electrolyte interface
# 120
ALLOWED_SLICES["IM_298-1_120"][1] = list(range(240,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_120"][2] = list(range(465,500)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_120"][3] = None # no film
ALLOWED_SLICES["IM_298-1_120"][4] = list(range(122,244)) + list(range(465,576)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_120"][5] = list(range(205,275)) # anode/electrolyte interface
# 130
ALLOWED_SLICES["IM_298-1_130"][1] = list(range(230,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_130"][2] = list(range(453,487)) # cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_130"][3] = list(range(218,275))
ALLOWED_SLICES["IM_298-1_130"][4] = list(range(118,245)) + list(range(453,564)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_130"][5] = list(range(212,250)) # anode/electrolyte interface
# 140
ALLOWED_SLICES["IM_298-1_140"][1] = list(range(220,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_140"][2] = list(range(215,246)) + list(range(430,500))  # anode/electrolyte and cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_140"][3] = list(range(220,280))
ALLOWED_SLICES["IM_298-1_140"][4] = list(range(123,250)) + list(range(440,575)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_140"][5] = list(range(215,275)) + list(range(360,500)) # anode/electrolyte and cathode/electrolyte interface
# 150
ALLOWED_SLICES["IM_298-1_150"][1] = list(range(220,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_150"][2] = list(range(215,246)) + list(range(430,500)) # anode/electrolyte and cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_150"][3] = list(range(222,282))
ALLOWED_SLICES["IM_298-1_150"][4] = list(range(123,250)) + list(range(438,576)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_150"][5] = list(range(218,255)) + list(range(360,495)) # anode/electrolyte and cathode/electrolyte interface
# 160
ALLOWED_SLICES["IM_298-1_160"][1] = list(range(220,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_160"][2] = list(range(214,255)) + list(range(360,505)) # anode/electrolyte and cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_160"][3] = list(range(215,285))
ALLOWED_SLICES["IM_298-1_160"][4] = list(range(120,255)) + list(range(441,584)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_160"][5] = list(range(210,285)) + list(range(360,505))  # anode/electrolyte and cathode/electrolyte interface
# 170
ALLOWED_SLICES["IM_298-1_170"][1] = list(range(215,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_170"][2] = list(range(215,253)) + list(range(430,495)) # anode/electrolyte and cathode/electrolyte interface
ALLOWED_SLICES["IM_298-1_170"][3] = list(range(230,280))
ALLOWED_SLICES["IM_298-1_170"][4] = list(range(120,253)) + list(range(445,577)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_170"][5] = list(range(215,280))  + list(range(355,500))  # anode/electrolyte and cathode/electrolyte interface
# 180
ALLOWED_SLICES["IM_298-1_180"][1] = list(range(213,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_180"][2] = list(range(213,250)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_180"][3] = list(range(230,275))
ALLOWED_SLICES["IM_298-1_180"][4] = list(range(118,250)) + list(range(450,574)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_180"][5] = list(range(213,275))  + list(range(355,495))  # anode/electrolyte and cathode/electrolyte interface
# 190
ALLOWED_SLICES["IM_298-1_190"][1] = list(range(214,365)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_190"][2] = list(range(214,251)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_190"][3] = list(range(233,277))
ALLOWED_SLICES["IM_298-1_190"][4] = list(range(120,251)) + list(range(455,580)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_190"][5] = list(range(214,277))  + list(range(360,500))  # anode/electrolyte and cathode/electrolyte interface
# 200
ALLOWED_SLICES["IM_298-1_200"][1] = list(range(214,360)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_200"][2] = list(range(214,254)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_200"][3] = list(range(234,277))
ALLOWED_SLICES["IM_298-1_200"][4] = list(range(120,254)) + list(range(453,582)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_200"][5] = list(range(214,277))  + list(range(360,503))  # anode/electrolyte and cathode/electrolyte interface
# 210
ALLOWED_SLICES["IM_298-1_210"][1] = list(range(213,360)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_210"][2] = list(range(213,250)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_210"][3] = None # no film
ALLOWED_SLICES["IM_298-1_210"][4] = list(range(123,250)) + list(range(462,583)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_210"][5] = list(range(213,260)) + list(range(360,505))  # anode/electrolyte and cathode/electrolyte interface
# 220
ALLOWED_SLICES["IM_298-1_220"][1] = list(range(217,310)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_220"][2] = list(range(217,246)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_220"][3] = None # no film
ALLOWED_SLICES["IM_298-1_220"][4] = list(range(125,247)) + list(range(460,579)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_220"][5] = list(range(217,246))  + list(range(375,501))  # anode/electrolyte and cathode/electrolyte interface
# 230
ALLOWED_SLICES["IM_298-1_230"][1] = list(range(218,310)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_230"][2] = list(range(218,236)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_230"][3] = None # no film
ALLOWED_SLICES["IM_298-1_230"][4] = list(range(126,236)) + list(range(466,573)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_230"][5] = list(range(218,236))  + list(range(375,500))  # anode/electrolyte and cathode/electrolyte interface
# 240
ALLOWED_SLICES["IM_298-1_240"][1] = list(range(218,310)) # anode half of electrolyte
ALLOWED_SLICES["IM_298-1_240"][2] = list(range(218,238)) # anode/electrolyte interface
ALLOWED_SLICES["IM_298-1_240"][3] = None # no film
ALLOWED_SLICES["IM_298-1_240"][4] = list(range(131,238)) + list(range(465,577)) # anode/cathode regions
ALLOWED_SLICES["IM_298-1_240"][5] = list(range(218,238))  + list(range(375,503))  # anode/electrolyte and cathode/electrolyte interface

# add deposited Li to film
for stack in ALLOWED_SLICES:
    if ALLOWED_SLICES[stack][3] is None:
        ALLOWED_SLICES[stack][3] = ALLOWED_SLICES[stack][5]
    else:
        ALLOWED_SLICES[stack][3] = ALLOWED_SLICES[stack][3] + ALLOWED_SLICES[stack][5]
