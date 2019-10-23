import numpy as np
from numpy import array


def hr_images(images):
    images_hr = array(images)
    return images_hr


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)