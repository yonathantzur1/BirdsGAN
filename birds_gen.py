import os

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

NOISE_SIZE = 256
PREVIEW_MARGIN = 1
IMAGE_SIZE = 128

PREVIEW_ROWS = 2
PREVIEW_COLS = 2
THRESHOLD = 0.9


GENERATOR_MODEL = "models/generator.h5"
DISCRIMINATOR_MODEL = "models/discriminator.h5"


def generate_image(generator, discriminator):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3), 255, dtype=np.uint8)

    noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
    generated_images = generator.predict(noise)

    while min(discriminator.predict(generated_images))[0] < THRESHOLD:
        noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
        generated_images = generator.predict(noise)

    print("acc: " + str(min(discriminator.predict(generated_images))[0]))
    generated_images = 0.5 * generated_images + 0.5
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    filename = os.path.join("./", "bird.png")
    im = Image.fromarray(image_array)
    im.save(filename)


generator = load_model(GENERATOR_MODEL)
discriminator = load_model(DISCRIMINATOR_MODEL)

print("Generating...")
generate_image(generator, discriminator)
print("done...")
