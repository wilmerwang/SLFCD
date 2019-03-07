import sys
import os
import argparse
import logging
import json

import keras

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))

from data.wsi_producer import WSIPatchDataset

parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                             'patch predictions given a WSI. ')
parser.add_argument('wsi_path', default=None, metavar='WSI PATH', type=str,
                    help='Path to the predict wsi')
parser.add_argument('mask_path', default=None, metavar='MASK PATH', type=str,
                    help='Path to the mask')
parser.add_argument('model_path', default=None, metavar='MODEL PATH', type=str,
                    help='Path to the trained model')
parser.add_argument('map_path', default=None, metavar='MAP PATH', type=str,
                    help='Path to the output probability map')


def get_probs_map(model, dataloder):
    probs_map = np.zeros(dataloder.dataset.mask.shape)
    steps = len(dataloder)

    for step in range(steps):
        data, xy_masks = next(dataloder)

        output = model.predict_on_batch(data)

        if len(output.shape) != 1:
            raise Exception('The model\'s len(output.shape) is not 1, it is'
                            '{}'.format(len(output.shape)))
        for i in range(len(output)):
            probs_map[xy_masks[i]] = output[i]
    return probs_map


def run(args):

    model = keras.models.load_model(args.model_path)

    dataset = WSIPatchDataset(args.wsi_path, 
                              args.mask_path, 
                              rescale=1./255)
    dataloader = dataset.patch_flow(target_size=256, batch_size=64)

    probs_map = get_probs_map(model, dataloader)

    np.save(args.map_path, probs_map)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
