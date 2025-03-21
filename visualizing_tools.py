import numpy as np
from random import randint
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def plot_two_images(image_1, image_2):
    fig = plt.figure(figsize=(6, 12))
    fig.add_subplot(2, 1, 1)
    plt.imshow(np.array(image_1).astype(np.uint8))
    fig.add_subplot(2, 1, 2)
    plt.imshow(np.array(image_2).astype(np.uint8))
    plt.show()
    plt.close()

def random_RGB():
    RGB = np.zeros(3, dtype=int)
    RGB[0] = randint(0, 255)
    RGB[1] = randint(0, 255)
    RGB[2] = randint(0, 255)
    return RGB

def RGB_Gaussian_filter(RGB_image, sigma):
    output = np.empty(shape=RGB_image.shape)
    output[:, :, 0] = gaussian_filter(RGB_image[:, :, 0], sigma=sigma)
    output[:, :, 1] = gaussian_filter(RGB_image[:, :, 1], sigma=sigma)
    output[:, :, 2] = gaussian_filter(RGB_image[:, :, 2], sigma=sigma)
    return output

def resize(PIL_image, target_pixels = 150000):
    original_width, original_height = PIL_image.size
    resize_coef = np.sqrt(target_pixels/(original_height * original_width))
    height = int(resize_coef * original_height)
    width = int(resize_coef * original_width)
    PIL_image = PIL_image.resize((width, height))
    return PIL_image

def segmentation_to_image(pixel_partition, height, width):
    output = np.zeros(shape=(height, width, 3))
    colors = np.zeros(shape=(height*width, 3))
    for i in range(width*height):
        colors[i, :] = random_RGB()
    for y in range(height):
        for x in range(width):
            rep = pixel_partition.find(y*width + x)
            output[y, x, :] = colors[rep, :]
    return output

def draw_bboxes(RGB_image, bboxes):
    output = np.array(RGB_image)
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        for p in range(x_min, x_max + 1):
            output[y_min, p, :] = [0, 0, 255]
            output[y_max, p, :] = [0, 0, 255]
        for p in range(y_min, y_max + 1):
            output[p, x_min, :] = [0, 0, 255]
            output[p, x_max, :] = [0, 0, 255]
    return output
