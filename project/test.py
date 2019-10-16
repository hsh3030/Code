import os.path 
import numpy as np 
from keras.models import * 
from keras.layers import * 
from keras.optimizers import * 
import keras.backend as K 
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')

class Gan:
    def __init__(self, img_data):
        img_size = img_data.shape[1]
        channel = img_data.shape[3] if len(img_data.shape) >= 4 else 1

        self.img_data = img_data
        self.input_shape = (img_size, img_size, channel)

        self.img_rows = img_size
        self.img_cols = img_size
        self.channel = channel
        self.noise_size = 100

        # Create D and G.
        self.create_d()
        self.create_g()

        # Build model to train D
        optimizer = Adam(lr=0.0008)
        self.D.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Build model to train G.
        optimizer = Adam(lr=0.0004)
        self.D.trainable = False
        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        self.AM.complie(loss='binary_crossentropy', optimizer=optimizer)

def create_d(self):
    self.D = Sequential()
    depth = 64
    dropout = 0.4
    self.D.add(Conv2D(depth * 1, 5, strides = 2, input_shape = self.input_shape, padding = 'same'))
    self.D.add(LeakyReLU(alpha=0.2))
    self.D.add(Dropout(dropout))
    self.D.add(Conv2D(depth * 2, 5, strides = 2, padding = 'same'))
    self.D.add(LeakyReLU(alpha=0.2))
    self.D.add(Dropout(dropout))
    self.D.add(Conv2D(depth * 4, 5, strides = 2, padding = 'same'))
    self.D.add(LeakyReLU(alpha=0.2))
    self.D.add(Dropout(dropout))
    self.D.add(Conv2D(depth * 8, 5, strides = 1, padding = 'same'))
    self.D.add(LeakyReLU(alpha=0.2))
    self.D.add(Dropout(dropout))
    self.D.add(Flatten())
    self.D.add(Dense(1))
    self.D.add(Activation('sigmoid'))
    self.D.summary()
    return self.D

def create_g(self):
    self.G = Sequential()
    dropout = 0.4
    depth = 64 + 64 + 64 + 64
    dim = 8
    self.G.add(Dense(dim * dim * depth, input_dim = self.noise_size))
    self.G.add(BatchNormalization(momentum=0.9))
    self.G.add(Activation('relu'))
    self.G.add(Reshape((dim, dim, depth)))
    self.G.add(Dropout(dropout))
    self.G.add(UpSampling2D())
    self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    self.G.add(BatchNormalization(momentum=0.9))
    self.G.add(Activation('relu'))
    self.G.add(UpSampling2D())
    self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    self.G.add(BatchNormalization(momentum=0.9))
    self.G.add(Activation('relu'))
    self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    self.G.add(BatchNormalization(momentum=0.9))
    self.G.add(Activation('relu'))
    self.G.add(Conv2DTranspose(self.channel, 5, padding='same'))
    self.G.add(Activation('sigmoid'))
    self.G.summary()
    return self.G

def train(self, batch_size=100):
    # Pick image data randomly.
    images_train = self.img_data[np.random.randint(0, self.img_data.shape[0], size=batch_size), :, :, :]

    # Generate images from noise.
    noise = np.random.uniform(-1.0, 1.0, size = [batch_size, self.noise_size])
    images_fake = self.G.predict(noise)

    # Train D.
    x = np.concatenate((image_train, images_fake))
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0
    self.D.trainable = True
    d_loss = self.D.train_on_batch(x, y)

    # Train G.
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size = [batch_size, self.noise_size])
    self.D.trainable = False
    a_loss = self.AM.train_on_batch(noise, y)

    return d_loss, a_loss, images_fake

def save(self):
    self.G.save_weights('gan_g_weights.h5')
    self.D.save_weights('gam_d_weights.h5')

def load(self):
    if os.path.isfile('gan_g_weights.h5'):
        self.G.load_weights('gam_g_weights.h5')
        print("Load G from file.")

    if os.path.isfile('gan_d_weights.h5'):
        self.D.load_weights('gan_d_weights.h5')
        print("Load D from file.")

class PokemonData():
    def __init__(self):
        img_data_list = []
        images = os.listdir("./setelite/train_images/00a47f4.jpg/")

        for path in images:
            img = Image.open("./setelite/train_images/00a47f4.jpg/" + path)
            img_data_list.append([np.array(img).astype('float32')])

        self.x_train = np.vstack(img_data_list) / 255.0
        print(self.x_train.shape)

# Load dataset.
dataset = PokemonData()

x_train = dataset.x_train

# Init network
gan = Gan(x_train)
gan.load()

# Some parameters
epochs = 1000
sample_size = 10
batch_size = 100
train_per_epoch = x_train.shape[0] // batch_size

for epoch in range(0, epochs):
    total_d_loss = 0.0
    total_a_loss = 0.0
    img = None

    for batch in range(0, train_per_epoch):
        d_loss, a_loss, t_imgs = gan.train(batch_size)
        total_d_loss += d_loss
        total_a_loss += a_loss
        if imgs is None:
            imgs = t_imgs

    if epoch % 20 == 0 or epoch == epochs - 1:
        total_d_loss /= train_per_epoch
        total_a_loss /= train_per_epoch

        print("Epoch: {}, D Loss: {}, AM Loss: {}".format(epoch, total_d_loss, total_a_loss))

        # Show generated images.
        fig, ax = plt.subplot(1, sample_size, figsize=(sample_size, 1))
        for i in range(0, sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(imgs[i].reshape((gan.img_rows, gan.img_cols, gan.channel)), interpolation='nearest');

        plt.show()
        plt.close(fig);

        # Save weights
        gan.save()