import os
import json
from argparse import ArgumentParser
import numpy as np
from skimage import io, util
import h5py
import os
import sys
import glob
import re
from tqdm import tqdm
import constants as const

import pdb


def crop_training_images(parser_args, cube):
    """Crops training images in smaller slices """
    # checking if folders exist.
    for folder in [prepare_args['crop_image_save_dir'], prepare_args['crop_label_save_dir']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # reading training images and their labels.
    data_folder = os.path.join(const.DATA_BASE,cube)
    label_folder = os.path.join(prepare_args['label_dir'],cube)
    if const.SAVE_BASE in label_folder:
        label_folder = os.path.join(label_folder,const.SUBFOLDER_LABEL)
    # label_folder = os.path.join(const.LABEL_BASE,cube)
    print(f"Loading slices in {cube}")
    data_idx = np.asarray([int(re.findall("img_([0-9]+).tif",x)[0]) for x in sorted(glob.glob(os.path.join(data_folder,f"img_*.tif")))])
    label_idx = np.asarray([int(re.findall("img_([0-9]+).tif",x)[0]) for x in sorted(glob.glob(os.path.join(label_folder,f"img_*.tif")))])
    vol = np.asarray([io.imread(os.path.join(data_folder,f"img_{i:04d}.tif")).astype(np.uint8) for i in data_idx])
    labels = np.asarray([io.imread(os.path.join(label_folder,f"img_{i:04d}.tif")).astype(np.uint8) for i in label_idx])
    if labels.shape[0] != vol.shape[0]:
        labels = labels.transpose(1,0,2)
    if labels.shape != vol.shape:
        print(f"vol is {vol.shape} but labes are {labels.shape} -- quitting ")
        return
    start = prepare_args['train_slices'][0]
    stop = prepare_args['train_slices'][1]
    slices = np.arange(start,stop+1)
    vol = vol[slices]
    labels = labels[slices]

    if const.SAVE_BASE in label_folder:
        labels = labels + 1 # add one to corresond to hand labels, where 0 is no label
    if prepare_args['orientation'] == 'cross-sectional':
        xz_ind = np.arange(0,vol.shape[1])
        yz_ind = np.arange(0,vol.shape[2])
        if "xsection_slices" in prepare_args:
            xz_ind = xz_ind[slice(*prepare_args["xsection_slices"])]
            yz_ind = yz_ind[slice(*prepare_args["xsection_slices"])]

        print(f"Cropping x-z cross-sectional slices")
        for i in tqdm(xz_ind):
            image, label = vol[:,i,:], labels[:,i,:]
            img_crop = np.vstack(util.view_as_windows(image,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))
            label_crop = np.vstack(util.view_as_windows(label,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))

            for j, (img, lab) in enumerate(zip(img_crop,label_crop)):
                if (np.count_nonzero(lab)/lab.size) > prepare_args['min_labeled']:
                    fname = f"{cube}_img_{i:04d}_crop-{j:04d}_xz.tif"
                    io.imsave(os.path.join(prepare_args['crop_image_save_dir'], fname), util.img_as_ubyte(img))
                    io.imsave(os.path.join(prepare_args['crop_label_save_dir'], fname), util.img_as_ubyte(lab),check_contrast=False)
        print(f"Cropping y-z cross-sectional slices")
        for i in tqdm(yz_ind):
            image, label = vol[:,:,i], labels[:,:,i]
            img_crop = np.vstack(util.view_as_windows(image,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))
            label_crop = np.vstack(util.view_as_windows(label,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))

            for j, (img, lab) in enumerate(zip(img_crop,label_crop)):
                if (np.count_nonzero(lab)/lab.size) > prepare_args['min_labeled']:
                    fname = f"{cube}_img_{i:04d}_crop-{j:04d}_yz.tif"
                    io.imsave(os.path.join(prepare_args['crop_image_save_dir'], fname), util.img_as_ubyte(img))
                    io.imsave(os.path.join(prepare_args['crop_label_save_dir'], fname), util.img_as_ubyte(lab),check_contrast=False)
    else: #in-plane
        print(f"Cropping x-y in-plane slices")
        for i in tqdm(range(vol.shape[0])):
            image, label = vol[i,:,:], labels[i,:,:]
            img_crop = np.vstack(util.view_as_windows(image,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))
            label_crop = np.vstack(util.view_as_windows(label,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size']))

            for j, (img, lab) in enumerate(zip(img_crop,label_crop)):
                if (np.count_nonzero(lab)/lab.size) > prepare_args['min_labeled']:
                    fname = f"{cube}_img_{i:04d}_crop-{j:04d}.tif"
                    io.imsave(os.path.join(prepare_args['crop_image_save_dir'], fname), util.img_as_ubyte(img))
                    io.imsave(os.path.join(prepare_args['crop_label_save_dir'], fname), util.img_as_ubyte(lab),check_contrast=False)

    return None


def crop_training_chunks(parser_args, cube):
    """Crops training and validation images in chunks.
    """

    # checking if folders exist.
    for folder in [prepare_args['crop_image_save_dir'], prepare_args['crop_label_save_dir']]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    # reading training images and their labels.
    data_folder = os.path.join(const.DATA_BASE,cube)
    label_folder = os.path.join(prepare_args['label_dir'],cube)
    if const.SAVE_BASE in label_folder:
        label_folder = os.path.join(label_folder,const.SUBFOLDER_LABEL)
    print(f"Loading slices in {cube}")
    data_idx = np.asarray([int(re.findall("img_([0-9]+).tif",x)[0]) for x in sorted(glob.glob(os.path.join(data_folder,f"img_*.tif")))])
    label_idx = np.asarray([int(re.findall("img_([0-9]+).tif",x)[0]) for x in sorted(glob.glob(os.path.join(label_folder,f"img_*.tif")))])
    vol = np.asarray([io.imread(os.path.join(data_folder,f"img_{i:04d}.tif")).astype(np.uint8) for i in data_idx])
    labels = np.asarray([io.imread(os.path.join(label_folder,f"img_{i:04d}.tif")).astype(np.uint8) for i in label_idx])
    if labels.shape[0] != vol.shape[0]:
        labels = labels.transpose(1,0,2)
    if labels.shape != vol.shape:
        print(f"vol is {vol.shape} but labes are {labels.shape} -- quitting ")
        return
    start = prepare_args['train_slices'][0]
    stop = prepare_args['train_slices'][1]
    slices = np.arange(start,stop+1)
    vol = vol[slices,slice(*prepare_args["xsection_slices"]),slice(*prepare_args["xsection_slices"])]
    labels = labels[slices,slice(*prepare_args["xsection_slices"]),slice(*prepare_args["xsection_slices"])]
    if const.SAVE_BASE in label_folder:
        labels = labels + 1 # add one to corresond to hand labels, where 0 is no label

    print(f"Cropping chunks of size {prepare_args['train_patch_size']}...")
    img_crop = util.view_as_windows(vol,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size'])
    label_crop = util.view_as_windows(labels,window_shape=prepare_args['train_patch_size'],step=prepare_args['step_size'])

    img_crop = img_crop.reshape(-1,*img_crop.shape[-3:])
    label_crop = label_crop.reshape(-1,*label_crop.shape[-3:])

    for j, (img, lab) in enumerate(zip(img_crop,label_crop)):
        if (np.count_nonzero(lab)/lab.size) > prepare_args['min_labeled']:
            fname = f"{cube}_crop-{j:04d}.tif"
            io.imsave(os.path.join(prepare_args['crop_image_save_dir'], fname), util.img_as_ubyte(img))
            io.imsave(os.path.join(prepare_args['crop_label_save_dir'], fname), util.img_as_ubyte(lab),check_contrast=False)

    return None

def create_hdf5_from_tif_stack(data_dir='//global/cfs/cdirs/als/users/ivzenyuk/07292021-NewBattery-298_1/prediction/pt_unet2d',stacks=const.STACKS):
    for stack in stacks:
        with h5py.File(os.path.join(const.DATA_BASE,stack+'.h5'),'r') as f:
            stack_shape = f['data'].shape
        files = sorted(glob.glob(os.path.join(data_dir,stack,const.SUBFOLDER_PRED,"img_*.tif")))
        idx = np.asarray([int(re.findall("img_([0-9]+).tif",x)[0]) for x in files])
        with h5py.File(os.path.join(data_dir,stack+'.h5'),'a') as f:
            f.create_dataset('data',stack_shape,dtype='u8')
            for i, file in zip(idx, files):
                f['data'][i] = io.imread(file)


def _parse_prepare_variables(argparse_args):
    """ Merges parameters from json config file and argparse, then parses/modifies parameters a bit"""
    prepare_args = vars(argparse_args)
    # overwrite argparse defaults with config file
    with open(prepare_args["config_file"]) as file_json:
        config_dict = json.load(file_json)
        prepare_args.update(config_dict)

    prepare_args['train_slices'] = tuple(prepare_args['train_slices']) # tuple expected, not list
    prepare_args['train_patch_size'] = tuple(prepare_args['train_patch_size']) # tuple expected, not list

    prepare_args['crop_image_save_dir'] = os.path.join(prepare_args['training_data_dir'],const.SUBFOLDER_IMAGE)
    prepare_args['crop_label_save_dir'] = os.path.join(prepare_args['training_data_dir'],const.SUBFOLDER_LABEL)

    return prepare_args

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    # add PROGRAM level args
    parser.add_argument("config_file", type=str, help="json file containing data and model training parameters")
    parser.add_argument("--label_dir", type=str, default=const.LABEL_BASE, help="folder where labels are")
    parser.add_argument("--training_data_dir", type=str, default=None, help="folder to save cropped images")
    parser.add_argument("--train_slices", type=int, default=(0,698), help="region to use for patching")
    parser.add_argument("--spatial_dims", type=int, default=2, help="crop 2D images or 3D chunks")
    parser.add_argument("--train_patch_size", type=int, default=(256,256), help="crop patch size")
    parser.add_argument("--step_size", type=int, default=(256,256), help="crop patch step size")
    parser.add_argument("--orientation", type=str, default='in-plane', help="crop orientation (in-plane or cross-sectional)")
    parser.add_argument("--min_labeled", type=float, default=0.2, help="minimum fraction of pixels that must be labeled to keep patch")
    args = parser.parse_args()

    prepare_args = _parse_prepare_variables(args)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # size, rank = 1, 0

    if prepare_args['spatial_dims'] == 3:
        for i, cube in enumerate(const.STACKS):
            if i % size == rank:
                crop_training_chunks(prepare_args,cube)
    else:
        for i, cube in enumerate(const.STACKS):
            if i % size == rank:
                crop_training_images(prepare_args,cube)
