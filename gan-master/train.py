from network.srgan import Generator as sr_generator
from network.esrgan import Generator as esr_generator
from network.complexgan import Generator as cpx_generator
from network.discriminator import Discriminator

import utils_model, utils
from utils_model import VGG_LOSS
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
import os
import argparse

# To fix error Initializing libiomp5.dylib
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(10)
# Better to use downscale factor as 4
downscale_factor = 2
# Remember to change image shape if you are having different size of images
image_shape = (120, 120, 3)


# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def get_gen_dis(model_type, image_shape, shape):
    if model_type == 'esr':
        generator = esr_generator(shape).generator()
    elif model_type == 'complex':
        generator = cpx_generator(shape).generator()
    else:
        generator = sr_generator(shape).generator()

    discriminator = Discriminator(image_shape).discriminator()

    return generator, discriminator


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(param_model_save, param_model_diff, epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, image_extension):
    # Loading images
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = \
        utils.load_training_data(input_dir, image_extension, image_shape, number_of_images, train_test_ratio)

    print('======= Loading VGG_loss ========')
    # Loading VGG loss
    loss = VGG_LOSS(image_shape)
    loss_diff = VGG_LOSS(image_shape)
    print('====== VGG_LOSS =======', loss)

    batch_count = int(x_train_hr.shape[0] / batch_size)
    print('====== Batch_count =======', batch_count)

    # shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])
    shape = image_shape     # 해상도만 낮추고 이미지 사이즈는 동일하게 처리
    print('====== Shape =======', shape)

    # Generator description, Discriminator description
    generator, discriminator = get_gen_dis(param_model_save, image_shape, shape)
    generator_diff, discriminator_diff = get_gen_dis(param_model_diff, image_shape, shape)

    optimizer = utils_model.get_optimizer()

    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    generator_diff.compile(loss=loss_diff.vgg_loss, optimizer=optimizer)

    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    discriminator_diff.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)
    gan_diff = get_gan_network(discriminator_diff, shape, generator_diff, optimizer, loss_diff.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')

    loss_file.close()

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images = generator.predict(image_batch_lr)
            generated_images_diff = generator_diff.predict(image_batch_lr)
            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True
            discriminator_diff.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            d_loss_real_diff = discriminator_diff.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake_diff = discriminator_diff.train_on_batch(generated_images_diff, fake_data_Y)
            discriminator_loss_diff = 0.5 * np.add(d_loss_fake_diff, d_loss_real_diff)
            #####
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            discriminator_diff.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])
            gan_loss_diff = gan_diff.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        print("gan_loss_diff :", gan_loss_diff)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt', 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
        loss_file.close()

        if e % 1 == 0:
            utils.plot_generated_images(output_dir, e, generator, generator_diff, x_test_hr, x_test_lr, [param_model_save, param_model_diff])
        if e % 50 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data_hr/',
#                         help='Path for input images')
#
#     parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
#                         help='Path for Output images')
#
#     parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
#                         help='Path for model')
#
#     parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=64,
#                         help='Batch Size', type=int)
#
#     parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=1000,
#                         help='number of iteratios for trainig', type=int)
#
#     parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1000,
#                         help='Number of Images', type=int)
#
#     parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8,
#                         help='Ratio of train and test Images', type=float)
#
#     values = parser.parse_args()
#
#     train(values.epochs, values.batch_size, values.input_dir, values.output_dir, values.model_save_dir,
#           values.number_of_images, values.train_test_ratio)

# Parameter
param_model_save = 'esr'
param_model_diff = 'complex'

param_epochs = 100#1000
param_batch = 5#10
param_input_folder = './VN_dataset/'
param_out_folder = './output/'
param_model_out_folder = './model/'
param_number_images = 50#500
param_train_test_ratio = 0.8
param_image_extension = '.png'

train(param_model_save,
        param_model_diff,
        param_epochs,
        param_batch,
        param_input_folder,
        param_out_folder,
        param_model_out_folder,
        param_number_images,
        param_train_test_ratio,
        param_image_extension)
