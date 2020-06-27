import os

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, \
    ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 50000  # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128  # rows/cols
IMAGE_CHANNELS = 3

GENERATOR_MODEL = "models/generator.h5"
DISCRIMINATOR_MODEL = "models/discriminator.h5"


def build_discriminator(image_shape):
    if DISCRIMINATOR_MODEL is not None:
        return load_model(DISCRIMINATOR_MODEL)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)


def build_generator(noise_size, channels):
    if GENERATOR_MODEL is not None:
        return load_model(GENERATOR_MODEL)

    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    for i in range(GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    input = Input(shape=(noise_size,))
    generated_image = model(input)

    return Model(input, generated_image)


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3), 255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    output_path = "output"
    create_dir(output_path)
    filename = os.path.join(output_path, "trained-" + str(cnt) + ".png")
    im = Image.fromarray(image_array)
    im.save(filename)


def save_models(cnt, generator_model, discriminator_model):
    output_path = "models"
    create_dir(output_path)
    output_path_dir = output_path + "/" + str(cnt)
    create_dir(output_path_dir)
    generator_model.save(output_path_dir + "/generator.h5")
    discriminator_model.save(output_path_dir + "/discriminator.h5")


# train #

image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = Adam(1.5e-4, 0.5)
discriminator = build_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
count = 0

training_data = np.load("birds_data.npy")

for epoch in range(EPOCHS):
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    x_real = training_data[idx]

    batch_noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
    x_fake = generator.predict(batch_noise)

    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
    generator_metric = combined.train_on_batch(batch_noise, y_real)

    if epoch % SAVE_FREQ == 0:
        count = count + 1
        print("epoch: " + str(epoch))
        print("Generator loss: " + str(generator_metric[0]) + ", Generator accuracy: " + str(100 * generator_metric[1]))
        print("Discriminator loss: " + str(discriminator_metric[0]) + ", Discriminator accuracy: " + str(
            100 * discriminator_metric[1]))
        print("-----------------------------------------------------------------")
        print()

        save_images(count, fixed_noise)
        save_models(count, generator, discriminator)
