import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL

class WSIPatchDataset(Dataset):

    def __init__(self, wsi_path, mask_path, image_size=256, crop_size=224,
                 normalize=True, flip='NONE', rotate='NONE'):
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._pre_process()

    def _pre_process(self):
        self._mask = np.load(self._mask_path)
        self._slide = openslide.OpenSlide(self._wsi_path)

        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape

        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match ,'
                            'X_slide / X_mask: {} / {}, '
                            'Y_slide / Y_mask: {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self._resolution = X_slide * 1.0 / X_mask
        if not np.log2(self._resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: '
                            '{}'.format(self._resolution))

        # all the idces for tissue region from the tissue mask
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)

        x = int(x_center - self._image_size / 2)
        y = int(y_mask - self._image_size / 2)

        img = self._slide.read_region(
            (x, y), 0, (self._image_size, self._image_size)).convert('RGB')

        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)

            # PIL image:   H x W x C
            # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return (img, x_mask, y_mask)