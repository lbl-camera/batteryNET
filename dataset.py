from abc import ABCMeta
import os
import sys
import glob
import random
import h5py
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset
from monai.data import CacheDataset
from monai.transforms import (
    MapTransform,
    AddChanneld,
    Compose,
    CastToTyped,
    MapLabelValued,
    SqueezeDimd,
    CenterSpatialCropd,
    RandFlipd,
    RandAffined,
    ScaleIntensityRanged,
    EnsureTyped
)
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
import constants as const

import pdb

class PreCroppedDataset(Dataset):
    def __init__(self, args, validation=False, idx=None):

        self.validation = validation
        self.training_data_dir = args['training_data_dir']
        self.data_fnames = sorted(glob.glob(os.path.join(self.training_data_dir,const.SUBFOLDER_IMAGE,'*'+const.EXT)))
        self.label_fnames = sorted(glob.glob(os.path.join(self.training_data_dir,const.SUBFOLDER_LABEL,'*'+const.EXT)))
        if len(self.data_fnames) != len(self.label_fnames):
             sys.exit(f"size of data and label images in {self.training_data_dir} are not the same")
        if not validation: # if training set, do random split of data
            idx = list(range(len(self.data_fnames)))
            random.shuffle(idx)
            self.valid_split = args['valid_split']
            split_ind = int(self.valid_split*len(idx))
            self.valid_idx = idx[:split_ind]
            self.idx = idx[split_ind:]
        else:
            if not isinstance(idx, list):
                sys.exit("Must input list of indices for validation set")
            self.idx = idx
        self.Nsamples = len(self.idx)
        self.data_dicts = [ {"image": self.data_fnames[idx], "label": self.label_fnames[idx]} for idx in self.idx ]

        self.train_patch_size = args["train_patch_size"]
        self.spatial_dims = len(self.train_patch_size)
        self.translate_range = args["translate_range"]
        self.rotate_dims = 1 if self.spatial_dims == 2 else 3
        self.flip_dims = (0,1) if self.spatial_dims == 2 else (0,1,2)
        self.rotate_range = np.pi*np.asarray(args["rotate_range"])
        if self.rotate_range.size != self.rotate_dims: self.rotate_range = self.rotate_range*np.ones(self.rotate_dims)

        self.scale_range = args["scale_range"]
        self.shear_range = args["shear_range"]
        self.use_cache = True if args["use_cache"] else False
        self.raw_labels = np.arange(0,args['num_classes']+1) if not args['raw_labels'] else args['raw_labels']
        self.output_labels = np.arange(-1,args['num_classes']) if not args['output_labels'] else args['output_labels']  # BatterySeg expectes unlabeled_index = -1, background_index = 0
        self.transform = self.get_data_transforms(validation)
        if self.use_cache:
            self.dataset = CacheDataset(
                data=self.data_dicts, transform=self.transform,
                cache_rate=1.0, num_workers=args["num_workers"])

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("precrop data")
        parser.add_argument("--training_data_dir", type=str, help="directory with pre-cropped training data")
        parser.add_argument("--valid_split", type=int, default=0.15, help="validation percentage (to split from train set)")
        parser.add_argument("--train_patch_size", type=int, default=[256,256], help="spatial size to output training patches (need not be same as input)")
        parser.add_argument("--translate_range", type=float, default=0.2, help="random translation for data augmentation (percent of patch size)")
        parser.add_argument("--rotate_range", type=float, default=1, help="random rotation for data augmentation (units of pi)")
        parser.add_argument("--scale_range", type=float, default=0.1, help="random scaling for data augmentation (from 0 to 1)")
        parser.add_argument("--shear_range", type=float, default=0.2, help="random shear for data augmentation (from 0 to 1)")
        parser.add_argument("--use_cache", type=bool, default=False, help="use MONAI cache dataset")
        parser.add_argument("--raw_labels", type=int, default=None, help="label indices for: (unlabeled,background,dendrite,pit,film)")
        parser.add_argument("--output_labels", type=int, default=None, help="label indices for: (unlabeled,background,dendrite,pit,film)")
        return parent_parser

    def get_valid_idx(self):
        if self.validation:
            return self.idx
        else:
            return self.valid_idx

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        if self.use_cache:
            return self.dataset[idx]
        else:
            return self.transform(self.data_dicts[idx])

    def get_data_transforms(self, validation):
        if validation:
            transform = Compose(
                [
                    tiff_reader(keys=["image","label"]),
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True,
                    ),
                    SqueezeDimd(keys=["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        else:
            transform = Compose(
                [
                    tiff_reader(keys=["image","label"]),
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True),
                    RandFlipd(
                            keys=['image', 'label'],
                            prob=0.5,
                            spatial_axis= self.flip_dims
                            ),
                    RandAffined(
                            keys=['image', 'label'],
                            mode=['bilinear','nearest'],
                            padding_mode='zeros', ### implicitly assumes raw unlabeled_index = 0!
                            prob=1.0,
                            spatial_size=self.train_patch_size,
                            rotate_range=self.rotate_range*np.ones(self.rotate_dims),
                            translate_range=self.translate_range*np.asarray(self.train_patch_size),
                            shear_range=self.shear_range*np.ones(self.spatial_dims),
                            scale_range=self.scale_range*np.ones(self.spatial_dims)
                            ),
                    SqueezeDimd(["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        return transform

class CropDataset3D(Dataset,metaclass=ABCMeta):
    def __init__(self, stack, args, validation=False):

        self.validation = validation
        self.training_data_dir = args["training_data_dir"]
        self.label_dir = args["label_dir"]
        self.stack = stack
        self.slices = args["training_slices"]
        self.data_fname = os.path.join(self.training_data_dir,self.stack+'.h5')
        self.label_fname = os.path.join(self.label_dir,self.stack+'.h5')
        self.data_files = h5py.File(self.data_fname,'r')
        self.label_files = h5py.File(self.label_fname,'r')
        self.data_shape = self.data_files['data'].shape
        self.label_shape = self.label_files['data'].shape
        if self.data_shape != self.label_shape:
            sys.exit(f"Size of {stack} data is {self.data_shape} but label is {self.label_shape} -- must be the same")
        self.Nsamples = args["num_val_images"] if self.validation else args["num_train_images"]

        self.spatial_dims = 3 # hardcoded
        self.rotate_dims = 3 # hardcoded
        self.train_patch_size = args["train_patch_size"]
        # set random crop size 50% bigger than desired (to avoid padded border at edge from transforms)
        self.crop_size = tuple(np.round(1.5*np.asarray(self.train_patch_size)).astype(int))
        self.data_buff = np.zeros(self.crop_size,dtype=np.uint8)
        self.label_buff = np.zeros(self.crop_size,dtype=np.uint8)
        self.rotate_range = np.pi*np.asarray(args["rotate_range"])
        if self.rotate_range.size != self.rotate_dims: self.rotate_range = self.rotate_range*np.ones(self.rotate_dims)
        self.scale_range = args["scale_range"]
        self.shear_range = args["shear_range"]
        self.raw_labels = np.arange(0,args['num_classes']+1) if not args['raw_labels'] else args['raw_labels']
        self.output_labels = np.arange(-1,args['num_classes']) if not args['output_labels'] else args['output_labels']  # BatterySeg expectes unlabeled_index = -1, background_index = 0
        self.transform = self.get_data_transforms(validation)

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("randcrop data")
        parser.add_argument("--training_data_dir", type=str, default=const.DATA_BASE, help="directory with hdf5 files of data")
        parser.add_argument("--label_dir", type=str, default=const.LABEL_BASE, help="directory with hdf5 files of labels")
        parser.add_argument("--training_stacks", type=list, default=const.STACKS, help="stack to use in training")
        parser.add_argument("--training_slices", type=list, default=[210,311], help="slices to use in dataset")
        parser.add_argument("--num_train_images", type=int, default=const.TRAIN_IMAGES_PER_EPOCH, help="number of images per epoch for training")
        parser.add_argument("--num_val_images", type=int, default=const.VAL_IMAGES_PER_EPOCH, help="number of images per epoch for validation")
        parser.add_argument("--train_patch_size", type=int, default=[16, 128,128], help="spatial size to output training patches (need not be same as input)")
        parser.add_argument("--rotate_range", type=float, default=1, help="random rotation for data augmentation (units of pi)")
        parser.add_argument("--scale_range", type=float, default=0.1, help="random scaling for data augmentation (from 0 to 1)")
        parser.add_argument("--shear_range", type=float, default=0.2, help="random shear for data augmentation (from 0 to 1)")
        parser.add_argument("--raw_labels", type=int, default=None, help="label indices for: (unlabeled,background,dendrite,pit,film)")
        parser.add_argument("--output_labels", type=int, default=None, help="label indices for: (unlabeled,background,dendrite,pit,film)")
        return parent_parser

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        pass

    def close_all_h5_files(self):
        [f.close() for f in self.data_files]
        [f.close() for f in self.label_files]

    def get_data_transforms(self, validation):
        if validation:
            transform = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True,
                    ),
                    CenterSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.train_patch_size),
                    SqueezeDimd(keys=["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        else:
            transform = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True),
                    RandAffined(
                            keys=['image', 'label'],
                            mode=['bilinear','nearest'],
                            padding_mode='zeros', ### implicitly assumes raw unlabeled_index = 0!
                            prob=1.0,
                            spatial_size=self.train_patch_size,
                            rotate_range=self.rotate_range,
                            shear_range=self.shear_range*np.ones(self.spatial_dims),
                            scale_range=self.scale_range*np.ones(self.spatial_dims)
                            ),
                    CenterSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.train_patch_size),
                    SqueezeDimd(["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        return transform

class RandomCropDataset3D(CropDataset3D):
    def __init__(self, stack, args, validation=False):
        super().__init__(stack, args, validation)

    def __getitem__(self, idx):
        zc = np.random.randint(low=self.slices[0], high=self.slices[1]-self.crop_size[0])
        xc = np.random.randint(low=0, high=self.data_shape[1]-self.crop_size[1])
        yc = np.random.randint(low=0, high=self.data_shape[2]-self.crop_size[2])
        window = tuple(slice(s, s + self.crop_size[i]) for i, s in enumerate([zc, xc, yc]))
        self.data_files['data'].read_direct(self.data_buff,window)
        self.label_files['data'].read_direct(self.label_buff,window)

        return self.transform({"image": self.data_buff, "label": self.label_buff})

class GridCropDataset3D(CropDataset3D):
    def __init__(self, stack, args, validation=False, idx=None):
        super().__init__(stack, args, validation)
        self.windows = dense_patch_slices(self.data_shape, self.crop_size, self.crop_size)
        if not validation: # if training set, do random split of data
            idx = list(range(len(self.windows)))
            random.shuffle(idx)
            self.valid_split = args['valid_split']
            split_ind = int(self.valid_split*len(idx))
            self.valid_idx = idx[:split_ind]
            self.idx = idx[split_ind:]
        else:
            if not isinstance(idx, list):
                sys.exit("Must input list of indices for validation set")
            self.idx = idx
        self.adjusted_windows = self.windows
        # self.jitter_windows()

    def __len__(self):
        return min(self.Nsamples,len(self.windows))

    def __getitem__(self, idx):
        window = self.windows[idx]
        self.data_files['data'].read_direct(self.data_buff,window)
        self.label_files['data'].read_direct(self.label_buff,window)
        return self.transform({"image": self.data_buff, "label": self.label_buff})

    def get_valid_idx(self):
        if self.validation:
            return self.idx
        else:
            return self.valid_idx

    def jitter_windows(self):
        z_jit = np.random.randint(low=0, high=self.crop_size[0]//2)
        x_jit = np.random.randint(low=0, high=self.crop_size[1]//2)
        y_jit = np.random.randint(low=0, high=self.crop_size[2]//2)
        adjusted = []
        for idx in self.idx:
            win = self.windows[idx]
            zslice, xslice, yslice = win
            if zslice.stop + z_jit < self.slices[1]-self.slices[0] and xslice.stop + x_jit < self.data_shape[1] and yslice.stop + y_jit < self.data_shape[2]:
                # add jitter to all dimensions. Also adjust all z slices to start at first_slice, not 0
                new_zslice = slice( zslice.start + self.slices[0] + z_jit, zslice.stop + self.slices[0] + z_jit, zslice.step)
                new_xslice = slice( xslice.start + x_jit, xslice.stop + x_jit, xslice.step)
                new_yslice = slice( yslice.start + y_jit, yslice.stop + y_jit, yslice.step)
                adjusted.append( (new_zslice, new_xslice, new_yslice) )
            else:
                adjusted.append( win )
        self.adjusted_windows = random.shuffle(adjusted)

    def close_all_h5_files(self):
        [f.close() for f in self.data_files]
        [f.close() for f in self.label_files]

    def get_data_transforms(self, validation):
        if validation:
            transform = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True,
                    ),
                    CenterSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.train_patch_size),
                    SqueezeDimd(keys=["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        else:
            transform = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=0, a_max=255,
                        b_min=0.0, b_max=1.0, clip=True),
                    RandAffined(
                            keys=['image', 'label'],
                            mode=['bilinear','nearest'],
                            padding_mode='zeros', ### implicitly assumes raw unlabeled_index = 0!
                            prob=1.0,
                            spatial_size=self.train_patch_size,
                            rotate_range=self.rotate_range,
                            shear_range=self.shear_range*np.ones(self.spatial_dims),
                            scale_range=self.scale_range*np.ones(self.spatial_dims)
                            ),
                    CenterSpatialCropd(
                        keys=['image', 'label'],
                        roi_size=self.train_patch_size),
                    SqueezeDimd(["label"],dim=0),
                    MapLabelValued(["label"],
                                   self.raw_labels,
                                   self.output_labels),
                    CastToTyped(keys=["label"],dtype=torch.long),
                    EnsureTyped(keys=["image", "label"])
                ]
            )

        return transform

# loading from hdf5 files
class StackDataset2D(Dataset):
    def __init__(self, stack, slices, args):
        self.pred_data_dir = args["pred_data_dir"]
        self.stack = stack
        self.data_file = os.path.join(self.pred_data_dir,stack+'.h5')
        with h5py.File(self.data_file,'r') as f:
            self.data_shape = f['data'].shape
        self.orientation = args['orientation']
        if slices:
            self.slices = range(slices[0],slices[-1])
        else:
            self.slices = range(0,self.data_shape[1]) if self.orientation == "cross-sectional" else range(0,self.data_shape[0])
        self.Nsamples = len(self.slices)

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("stack2d data")
        parser.add_argument("--pred_data_dir", type=str, default=const.DATA_BASE, help="directory with hdf5 files of data")
        parser.add_argument("--pred_patch_size", type=int, default=[1024,1024], help="spatial size of input prediction patches")
        parser.add_argument("--orientation", type=str, default='in-plane', help="orientation of slices: in-plane, or cross-sectional")
        return parent_parser

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        filename = f"img_{self.slices[idx]:04d}.tif"
        if self.orientation == "cross-sectional":
            with h5py.File(self.data_file,'r') as f:
                img = f['data'][:,self.slices[idx],:]
        else:
            with h5py.File(self.data_file,'r') as f:
                img = f['data'][self.slices[idx],:,:]
        img = torch.unsqueeze(torch.from_numpy(img),0).float()/255.0
        return (img, filename)

# loading from tif files
# class StackDataset2D(Dataset):
#     def __init__(self, stack, slices, args):
#         self.pred_data_dir = args["pred_data_dir"]
#         self.stack = stack
#         self.slices = slices
#         self.files = [f"img_{i:04d}.tif" for i in self.slices]
#         self.orientation = args['orientation']
#         self.Nsamples = len(self.files)
#         self.shape = io.imread(os.path.join(self.pred_data_dir,self.stack,self.files[0])).shape
#
#     @staticmethod
#     def add_args(parent_parser):
#         parser = parent_parser.add_argument_group("stack2d data")
#         parser.add_argument("--pred_data_dir", type=str, default=const.DATA_BASE, help="directory with hdf5 files of data")
#         parser.add_argument("--pred_patch_size", type=int, default=[1024,1024], help="spatial size of input prediction patches")
#         parser.add_argument("--orientation", type=str, default='in-plane', help="orientation of slices: in-plane, or cross-sectional")
#         return parent_parser
#
#     def __len__(self):
#         if self.orientation == 'cross-sectional':
#             return self.shape[0]
#         else:
#             return self.Nsamples
#
#     def __getitem__(self, idx):
#         filename = self.files[idx]
#         img = io.imread(os.path.join(self.pred_data_dir,self.stack,filename))/255.0
#         img = torch.unsqueeze(torch.from_numpy(img),0).float()
#         return (img, filename)

class StackDataset3D(Dataset):
    def __init__(self, stack, slices, args):

        self.pred_data_dir = args["pred_data_dir"]
        self.stack = stack
        self.slices = slices
        self.first_slice = self.slices[0]
        self.data_file = os.path.join(self.pred_data_dir,stack+'.h5')
        with h5py.File(self.data_file,'r') as f:
            self.data_shape = (len(self.slices), *f['data'].shape[1:])

        self.spatial_dims = 3 # hardcoded
        self.pred_patch_size = args["pred_patch_size"]
        self.overlap = args["overlap"]
        scan_interval = self._get_scan_interval()
        # Store all slices in list
        self.windows = dense_patch_slices(self.data_shape, self.pred_patch_size, scan_interval)
        self.adjusted_windows = self._adjust_windows_to_first_slice()
        self.Nsamples = len(self.windows)  # number of windows per image
        # Create window-level importance map
        self.importance_map = compute_importance_map(
            get_valid_patch_size(self.data_shape, self.pred_patch_size), mode='constant')

        self.data_buff = np.zeros(self.pred_patch_size,dtype=np.uint8)

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("stack3d data")
        parser.add_argument("--pred_data_dir", type=str, default=const.DATA_BASE, help="directory with hdf5 files of data")
        parser.add_argument("--pred_patch_size", type=int, default=[32,256,256], help="spatial size of rolling window prediction patch")
        parser.add_argument("--overlap", type=float, default=0.25, help="prediction patch overlap percentage (0-to-1)")
        return parent_parser

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        window = self.adjusted_windows[idx]
        with h5py.File(self.data_file,'r') as f:
            f['data'].read_direct(self.data_buff,window)
            chunk = torch.unsqueeze(torch.from_numpy(self.data_buff/255.0),0).float()
        return (chunk, idx)

    def _adjust_windows_to_first_slice(self):
        adjusted = []
        for i, win in enumerate(self.windows):
            zslice, xslice, yslice = win
            # adjust all z slices to start at self.first_slice, not 0
            new_zslice = slice( zslice.start+self.first_slice,
                                zslice.stop+self.first_slice,
                                zslice.step)
            adjusted.append( (new_zslice, xslice, yslice) )
        return adjusted

    def aggregate_patches(self, patches, idx):
        output_channels = patches[0].shape[0]
        output_shape = (output_channels,*self.data_shape)
        output_image = torch.zeros(output_shape, dtype=torch.float32)
        count_map = torch.zeros(output_shape, dtype=torch.uint8)
        # store the result in the proper location of the full output. Apply weights from importance map.
        for i, patch in zip(idx, patches):
            for j in range(output_channels):
                output_image[j][self.windows[i]] += self.importance_map * patch[j]
                count_map[j][self.windows[i]] += self.importance_map

        # account for any overlapping sections
        output_image = output_image / count_map
        return output_image

    def _get_scan_interval(self):
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.
        """
        if len(self.data_shape) != self.spatial_dims:
            raise ValueError("image coord different from spatial dims.")
        if len(self.pred_patch_size) != self.spatial_dims:
            raise ValueError("roi coord different from spatial dims.")

        scan_interval = []
        for i in range(self.spatial_dims):
            if self.pred_patch_size[i] == self.data_shape[i]:
                scan_interval.append(int(self.pred_patch_size[i]))
            else:
                interval = int(self.pred_patch_size[i] * (1 - self.overlap))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

class tiff_reader(MapTransform):
    def __init__(self,keys=["image","label"]):
        self.keys = keys
        MapTransform.__init__(self, self.keys, allow_missing_keys=False)

    def __call__(self, data_dict):
        data = {}
        for key in self.keys:
            if key in data_dict:
                data[key] = io.imread(data_dict[key])
        return data
