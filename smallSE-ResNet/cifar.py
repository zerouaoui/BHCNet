import os
import math
import argparse
import tensorflow as tf
import keras
from keras import optimizers
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

from model.ResNet_cifar import *
from utils.misc import color_preprocessing, mkdir_p

parser = argparse.ArgumentParser(description='ResNet')
parser.add_argument('--resnet', type=str, default='resnet_18', choices=['resnet_18', 'resnet_50'], help='The model name.')
parser.add_argument('--is_se', action='store_true', default=False, help='Is it SENet? Default: False.')
parser.add_argument('--block', type=str, default='basic_block',
                    choices=['basic_block', 'small_basic_block', 'bottleneck'],
                    help='ResNet block type. Default: basic_block.')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'], help='The dataset name.')
parser.add_argument('--batch_size', type=int, default=128, help='Input batch size for training. Default: 128.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train. Default: 200.')
# gpu device
parser.add_argument('--device', type=str, default='0', help='GPU device. Default: 0.')
parser.add_argument('--allow_growth', action='store_true', default=True, help='GPU memory.')

args = parser.parse_args()

def step(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
config = tf.ConfigProto()
config.gpu_options.allow_growth = args.allow_growth
sess = tf.Session(config=config)

batch_size = args.batch_size
epochs = args.epochs
resnet = globals().get(args.resnet)
block = globals().get(args.block)

if __name__ == '__main__':
    if args.dataset == "cifar100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = color_preprocessing(x_train, x_test)

    is_se = '_se' if args.is_se else ''
    base_path = 'logs/' + args.dataset + '/' + args.resnet + is_se + '/' + args.block + '/'
    mkdir_p(base_path)
    index = str(len(os.listdir(base_path)) + 1)
    base_path = base_path + index + '/'
    mkdir_p(base_path + 'board/')
    mkdir_p(base_path + 'check/')

    print(args)

    if str(args.resnet).startswith('resnet_'):
        model = resnet((3, 32, 32), num_classes, block=block, is_se=args.is_se)
    else:
        model = resnet((32, 32, 3), num_classes)

    with open(base_path + 'args.txt', 'w') as f:
        for arg in vars(args):
            f.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    # print(model.summary())
    with open(base_path + 'model.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    max_acc = ModelCheckpoint(base_path + 'check/max_acc.h5', save_best_only=True, monitor='val_acc', verbose=1)
    min_loss = ModelCheckpoint(base_path + 'check/min_loss.h5', save_best_only=True, monitor='val_loss', verbose=1)

    cbks = [TensorBoard(log_dir=base_path + 'board/', histogram_freq=0),
            LearningRateScheduler(step),
            max_acc,
            min_loss]

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    # start training
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                               steps_per_epoch=math.ceil(x_train.shape[0] // batch_size),
                               epochs=epochs,
                               callbacks=cbks,
                               validation_data=(x_test, y_test))

    model.save(base_path + 'check/last.h5')

    with open(base_path + 'logs.txt', 'w') as f:
        f.write(str(hist.history))
