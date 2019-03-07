import numpy as np
import openslide

np.random.seed(0)


class WSIPatchDataset(object):

    def __init__(self,
                 wsi_path,
                 mask_path,
                 rescale=None):

        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.rescale = rescale

    def _preprocess(self):
        self._mask = np.load(self.mask_path)
        self._slide = openslide.OpenSlide(self.wsi_path)

        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask = self._mask.shape

        if X_slide / X_mask != Y_slide / Y_mask:
            raise Exception('Slide/Mask dimension does not match, '
                            'X_slide / X_mask: {} / {},'
                            'Y_slide / y_mask: {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self._resulution = X_slide * 1.0 / X_mask
        if not np.log2(self._resulution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2: {}'
                            .format(self._resulution))

        # all the idces for tissue region from the tissue mask
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)

    def patch_flow(self,
                   target_size=256,
                   batch_size=32):

        index = 0
        while True:
            img_flat = np.zeros((batch_size, target_size, target_size, 3),
                                dtype=np.float32)
            xy_masks = []
            for i in range(batch_size):
                x_mask, y_mask = self._X_idcs[index], self._Y_idcs[index]

                x_center = int(x_mask * self._resulution)
                y_center = int(y_mask * self._resulution)

                x = int(x_center - target_size / 2)
                y = int(y_center - target_size / 2)

                img = self._slide.read_region((x, y), 0,
                                              (target_size, target_size)).convert('RGB')

                # PIL image: H × W × C
                img = np.array(img, dtype=np.float32)
                if self.rescale:
                    img = img * self.rescale

                img_flat[i] = img
                xy_masks.append((x_mask, y_mask))

                index += 1
                index = index % self._idcs_num
                if index == 0:
                    index = 0

            yield (img_flat, xy_masks)
