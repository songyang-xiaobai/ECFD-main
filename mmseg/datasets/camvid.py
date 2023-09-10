import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamvidsDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.

    """

    CLASSES = ('sky', 'building', 'pole', 'road', 'sidewalk', 'tree',
               'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist')

    PALETTE = [[128, 128, 128], [128, 0, 0], [192, 192, 192], [128, 64, 128],
               [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
               [64, 0, 128], [64, 64, 0], [0, 128, 192]]
    # palette = [128, 38, 185, 90, 22, 113, 147, 71, 34, 57, 97]  # background 0

    def __init__(self, **kwargs):
        super(CamvidsDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_L.png',
            **kwargs)