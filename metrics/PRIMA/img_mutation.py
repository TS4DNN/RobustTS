import numpy as np


def generate_ratio_vector(num, ratio):
    import math
    perturbate_num = math.ceil(num * ratio)
    non_perturbate_num = num - perturbate_num
    a = np.zeros(perturbate_num)+1
    b = np.zeros(non_perturbate_num)
    a_b = np.concatenate((a, b), axis=0)
    np.random.shuffle(a_b)
    return a_b


def black(image, i=0, j=0):
    image = np.array(image, dtype=float)
    image[0+2*i:2+2*i, 0+2*j:2+2*j] = 0
    return image.copy()


def white(image,i=0,j=0):
    image = np.array(image, dtype=float)
    image[0+2*i:2+2*i, 0+2*j:2+2*j] = 255
    return image.copy()


def reverse_color(image, i=0, j=0):
    image = np.array(image, dtype=float)
    part = image[0+2*i:2+2*i, 0+2*j:2+2*j].copy()
    reversed_part = 255-part
    image[0+2*i:2+2*i, 0+2*j:2+2*j] = reversed_part
    return image


def gauss_noise(image, i=0, j=0, mean=0, var=0.1, ratio=1.0):
    image = np.array(image, dtype=float)
    image = image.astype('float32') / 255
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    ratio_vector = generate_ratio_vector(len(part.ravel()), ratio).reshape(part.shape)
    noise = np.random.normal(mean, var ** 0.5, part.shape)
    noise = noise * ratio_vector
    image[0+2*i:2+2*i,0+2*j:2+2*j] += noise
    image = np.clip(image, 0, 1)
    image *= 255
    return image.copy()


def shuffle_pixel(image, i=0, j=0):
    image = np.array(image, dtype=float)
    # image /= 255
    part = image[0+2*i:2+2*i, 0+2*j:2+2*j].copy()
    part_r = part.reshape(-1, 1)
    np.random.shuffle(part_r)
    part_r = part_r.reshape(part.shape)
    image[0+2*i:2+2*i, 0+2*j:2+2*j] = part_r
    return image
