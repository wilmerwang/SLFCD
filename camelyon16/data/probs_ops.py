import cv2
import numpy as np
import scipy.stats.stats as st

from skimage.measure import label
from skimage.measure import regionprops
from openslide import OpenSlide
from openslide import OpenSlideUnsupportedFormatError

MAX, MEAN, VARIANCE, SKEWNESS, KURTOSIS = 0, 1, 2, 3, 4


class extractor_features(object):
    def __init__(self, probs_map, slide_path):
        self._probs_map = probs_map
        self._slide = get_image_open(slide_path)

    def get_region_props(self, probs_map_threshold):
        labeled_img = label(probs_map_threshold)
        return regionprops(labeled_img, intensity_image=self._probs_map)

    def probs_map_set_p(self, threshold):
        probs_map_threshold = np.array(self._probs_map)

        probs_map_threshold[probs_map_threshold < threshold] = 0
        probs_map_threshold[probs_map_threshold >= threshold] = 1

        return probs_map_threshold

    def get_num_probs_region(self, region_probs):
        return len(region_probs)

    def get_tumor_region_to_tissue_ratio(self, region_props):
        tissue_area = cv2.countNonZero(self._slide)
        tumor_area = 0

        n_regions = len(region_props)
        for index in range(n_regions):
            tumor_area += region_props[index]['area']

        return float(tumor_area) / tissue_area

    def get_largest_tumor_index(self, region_props):

        largest_tumor_index = -1

        largest_tumor_area = -1

        n_regions = len(region_props)
        for index in range(n_regions):
            if region_props[index]['area'] > largest_tumor_area:
                largest_tumor_area = region_props[index]['area']
                largest_tumor_index = index

        return largest_tumor_index

    def f_area_largest_tumor_region_t50(self):
        pass

    def get_longest_axis_in_largest_tumor_region(self,
                                                 region_props,
                                                 largest_tumor_region_index):
        largest_tumor_region = region_props[largest_tumor_region_index]
        return max(largest_tumor_region['major_axis_length'],
                   largest_tumor_region['minor_axis_length'])

    def get_average_prediction_across_tumor_regions(self, region_props):
        # close 255
        region_mean_intensity = [region.mean_intensity for region in region_props]
        return np.mean(region_mean_intensity)

    def get_feature(self, region_props, n_region, feature_name):
        feature = [0] * 5
        if n_region > 0:
            feature_values = [region[feature_name] for region in region_props]
            feature[MAX] = format_2f(np.max(feature_values))
            feature[MEAN] = format_2f(np.mean(feature_values))
            feature[VARIANCE] = format_2f(np.var(feature_values))
            feature[SKEWNESS] = format_2f(st.skew(np.array(feature_values)))
            feature[KURTOSIS] = format_2f(st.kurtosis(np.array(feature_values)))

        return feature


def format_2f(number):
    return float("{0:.2f}".format(number))


def get_image_open(wsi_path):
    try:
        wsi_image = OpenSlide(wsi_path)
        level_used = wsi_image.level_count - 1
        rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
        wsi_image.close()
    except OpenSlideUnsupportedFormatError:
        raise ValueError('Exception: OpenSlideUnsupportedFormatError for %s' % wsi_path)

    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)

    return image_open