import sys
import os
import argparse
import logging
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import inception_v3, \
    inception_resnet_v2, resnet50, densenet
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))

parser = argparse.ArgumentParser(description='Train the base model. ')
parser.add_argument('cnn_path', default=None, metavar='CNN NAME', type=str,
                    help='Path to the config file in json format. ')
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the saved model. ')


def my_model(cnn):
    base_model = cnn['model'](weights='imagenet',
                              include_top=False,
                              pooling='avg')
    x = base_model.output
    x = Dense(1664, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model, outputs=prediction)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def callback(args):
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(args.save_path, 'train.h5'),
                                       monitor='val_loss', save_best_only=True, save_weights_only=False,
                                       period=1)
    csv_logger = CSVLogger(filename=os.path.join(args.save_path, 'result.csv'),
                           separator=',', append=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                  min_lr=0.00001)
    tbcallback = TensorBoard(log_dir=os.path.join(args.save_path, 'log/'),
                             histogram_freq=0, batch_size=args.batch_size,
                             write_graph=False, write_images=True, update_freq='epoch')
    return [model_checkpoint, reduce_lr, csv_logger, tbcallback]


def run(args):
    with open(args.cnn_path) as f:
        cnn = json.load(f)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'cnn.json'), 'w') as f:
        json.dump(cnn, f, indent=1)

    model = my_model(cnn=cnn)

    # 应当对图片进行处理 # 标签
    dataset_train = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       horizontal_flip=True)
    dataset_valid = ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=False)
    dataloader_train = dataset_train.flow_from_directory(cnn['data_path_train'],
                                                         target_size=(256, 256),
                                                         batch_size=cnn['batch_size'],
                                                         class_mode='categorical',
                                                         classes={'normal': 0, 'tumor': 1})
    dataloader_valid = dataset_valid.flow_from_directory(cnn['data_path_valid'],
                                                         target_size=(256, 256),
                                                         batch_size=cnn['batch_size'] * 2,
                                                         class_mode='categorical',
                                                         classes={'normal': 0, 'tumor': 1})

    model.fit_generator(dataloader_train, epochs=cnn['epoch'], verbose=1,
                        callbacks=callback(args), validation_data=dataloader_valid,
                        workers=6, shuffle=True, initial_epoch=0)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

