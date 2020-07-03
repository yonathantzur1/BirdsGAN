import os
import time

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

NOISE_SIZE = 256
IMAGE_SIZE = 128

GENERATE_PATH = "gan_images"
THRESHOLD = 0.98
AMOUNT = 5

GENERATOR_MODEL = "models/generator.h5"
DISCRIMINATOR_MODEL = "models/discriminator.h5"


def get_time():
    return int(round(time.time() * 1000))


def generate_image(generator, discriminator):
    noise = np.random.normal(0, 1, (AMOUNT, NOISE_SIZE))
    generated_images = generator.predict(noise)
    predict = discriminator.predict(generated_images)

    while min(predict) < THRESHOLD:
        false_indexes = []
        for i in range(len(predict)):
            if predict[i] < THRESHOLD:
                false_indexes.append(i)
        noise = np.random.normal(0, 1, (len(false_indexes), NOISE_SIZE))
        fixed_images = generator.predict(noise)
        for i in range(len(false_indexes)):
            generated_images[false_indexes[i]] = fixed_images[i]
        predict = discriminator.predict(generated_images)

    print("Minimum accuracy: " + str(min(predict)[0]))
    generated_images = 0.5 * generated_images + 0.5
    for index in range(AMOUNT):
        image_array = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        image_array[0:0 + IMAGE_SIZE, 0:0 + IMAGE_SIZE] = generated_images[index] * 255
        filename = os.path.join(GENERATE_PATH, str(get_time()) + ".png")
        im = Image.fromarray(image_array)
        im.save(filename)


generator = load_model(GENERATOR_MODEL)
discriminator = load_model(DISCRIMINATOR_MODEL)

print("Generating...")
generate_image(generator, discriminator)
print("done...")
