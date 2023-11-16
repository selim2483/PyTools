"""Streaming images and labels from datasets created with dataset_tool.py."""

import argparse
import json
import os
from typing import Union, override
import zipfile

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

from pytools.utils.misc import EasyDict
from pytools.data.augmentations import augment

__all__ = ["Dataset", "ImageFolderDataset"]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

class Dataset(torch.utils.data.Dataset):
    """Base class for datasets."""

    def __init__(self,
        name        :str,
        raw_len     :int,
        resolution  :int,
        max_size    :Union[int,None] = None,
        use_labels  :bool            = False,
        random_seed :int             = 0,
        augment     :bool            = False,
        transform   :torch.nn.Module = DEFAULT_TRANSFORM,
        **kwargs
        ):
        """
        Args:
            name (str): Name of the dataset.
            raw_len (str): Number of images in dataset.
            resolution (int): Wanted resolution for images.
            max_size (int|None, optional): Artificially limit the size of the
                dataset. 
                Default to `None` for no limitation.
            use_labels (bool, optional): Enable conditioning labels
                If `False` : label dimension is zero.
                Default to `False`
            random_seed (int, optional): Random seed to use when applying
                `max_size`
                Default to `0`.
            augment (bool, optional): Enable augmentations (random resized
                crop and rotation).
                Default to `False`.
            transform (torch.nn.Module, optional): torchvision transforms to
                be applied on images. 
                Defaults to DEFAULT_TRANSFORM (ToTensor and Normalize to [0,1]
                range).
        """
        self._name        = name
        self._raw_len     = raw_len
        self.resolution   = resolution
        self._use_labels  = use_labels
        self._raw_labels  = None
        self._label_shape = None
        self.augment      = augment
        self.transform    = transform
        if self.augment :
            self.min_scale = kwargs.get("min_scale", 0.3)
            self.max_scale = kwargs.get("max_scale", 1.)
            self.min_theta = kwargs.get("min_theta", 0.)
            self.max_theta = kwargs.get("max_theta", 2 * np.pi)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_len, dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_len, 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0]==self._raw_len
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype==np.int64:
                assert self._raw_labels.ndim==1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass
    
    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert image.dtype==np.uint8

        image = self.transform(image)
        
        if self.augment :
            scale = (torch.rand(1) 
                    * (self.max_scale - self.min_scale) + self.min_scale)
            theta = (torch.rand(1) 
                    * (self.max_theta - self.min_theta) + self.min_theta)
            image = augment(image, scale, theta, self.resolution)

        return image.squeeze(0), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype==np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype==np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape)==1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype==np.int64


class ImageFolderDataset(Dataset):
    """Dataset class for folder datasets : either directory or `.zip` file."""
    def __init__(self,
        path :str,
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
        ):
        """
        Args:
            path (str): Path to directory or zip.
            resolution (int): 
        Raises:
            IOError: _description_
            IOError: _description_
        """
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path) 
                for root, _dirs, files in os.walk(self._path) 
                for fname in files
            }
        elif self._file_ext(self._path)=='.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(
            fname 
            for fname in self._all_fnames 
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames)==0:
            raise IOError('No image files found in the specified path')

        super().__init__(
            name       = os.path.splitext(os.path.basename(self._path))[0], 
            raw_len    = len(self._image_fnames), 
            resolution = resolution, 
            **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type=='zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type=='dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type=='zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f))
        if image.ndim==2:
            image = image[:, :, np.newaxis].repeat(3, axis=2) # HW => HWC
        image = image # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
            self, 
            dataset, 
            rank         = 0, 
            num_replicas = 1, 
            shuffle      = True, 
            seed         = 0, 
            window_size  = 0.5
        ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas==self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=str, 
        default="/scratchm/sollivie/data/MacroTextures500/train.zip"
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--use_labels", type=bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--augment", type=bool, default=False)

    args = parser.parse_args()
    print(vars(args))

    if os.path.isdir(args.path):
        data = ImageFolderDataset(**vars(args))
    elif os.path.splitext(args.path)[1].lower()=='.zip' :
        data = ImageFolderDataset(**vars(args))
    else :
        raise IOError('Path must point to a directory or zip')
    
    img, label = data[0]
    print(img.shape)
    print(label)

    img, label = data[3]
    print(img.shape)
    print(label)

    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=4)
    print(loader.__class__.__name__)
    print(type(loader))
    print(type(iter(loader)))

