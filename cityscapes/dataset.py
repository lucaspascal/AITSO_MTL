"""
Directly modified from : https://github.com/meetps/pytorch-semseg
"""

import os
import torch
import numpy as np
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms as T
from PIL import Image



VALID_CLASSES = [
            [7, 8],
            [11, 12, 13],
            [17, 19, 20],
            [21, 22],
            [23],
            [24, 25],
            [26, 27, 28, 31, 32, 33],
        ]
CLASS_NAMES = [
            ["road", "sidewalk"],
            ["building", "wall", "fence"],
            ["pole", "traffic_light", "traffic_sign"],
            ["vegetation", "terrain"],
            ["sky"],
            ["person", "rider"],
            ["car", "truck", "bus", "train", "motorcycle", "bicycle"],
        ]

    
    
class Cityscapes(VisionDataset):
    """
    `Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    """

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            im_size: float = 128,
            target_type: Union[List[str], str] = "semantic",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.segs_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.images = []
        self.segs = []
        
        # Transforms
        self.img_transforms = T.Compose([
            T.Resize(im_size, interpolation=Image.BILINEAR), 
            T.ToTensor()])
        self.seg_transforms = T.Compose([
            T.Resize(im_size, interpolation=Image.NEAREST), 
        ])
        
        # Fills lists by looping over the different cities
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            seg_dir = os.path.join(self.segs_dir, city)
            for file_name in os.listdir(img_dir):
                seg_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             '{}_labelIds.png'.format(self.mode))
                    
                self.images.append(os.path.join(img_dir, file_name))
                self.segs.append(os.path.join(seg_dir, seg_name))
                

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        # Loads images
        img = Image.open(self.images[index]).convert('RGB')
        seg = Image.open(self.segs[index])
        

        # Transforms
        img = self.img_transforms(img)
        seg = np.array(self.seg_transforms(seg))
        
        # Set segmentation for binary classification
        h,w = seg.shape
        seg_masks = np.zeros((len(VALID_CLASSES), h, w), dtype=np.bool)
        for k in range(len(VALID_CLASSES)):
            seg_masks[k,:,:] = (sum([seg==elt for elt in VALID_CLASSES[k]])).astype(np.bool)
        seg_masks = torch.from_numpy(seg_masks).type(torch.float32)

        return img, seg_masks


    def __len__(self) -> int:
        return len(self.images)



