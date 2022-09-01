from __future__ import print_function
import sys
import cv2
import numpy as np
import random
import time
import copy
import tensorflow as tf
#
# sys.setdefaultencoding('utf8')


# keras 1.2.2 tf:1.2.0
class Mutators():
    def image_contrast(img_data, params):
        alpha = params
        new_img = cv2.multiply(img_data, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img_data, beta)                                  # new_img = img*alpha + beta

        return new_img

    def image_brightness(img_data, params):
        beta = params
        # new_img = cv2.add(self, img_data, beta)  # new_img = img*alpha + beta
        # return new_img
        img = tf.image.random_brightness(img_data, max_delta=beta)
        return img.numpy()

    def image_blur(img_data, params):
        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img_data, (3, 3))
        if params == 2:
            blur = cv2.blur(img_data, (4, 4))
        if params == 3:
            blur = cv2.blur(img_data, (5, 5))
        # if params == 4:
        #     blur = cv2.GaussianBlur(img_data, (3, 3), 0)
        # if params == 5:
        #     blur = cv2.GaussianBlur(img_data, (5, 5), 0)
        # if params == 6:
        #     blur = cv2.GaussianBlur(img_data, (7, 7), 0)
        # if params == 7:
        #     blur = cv2.medianBlur(img_data, 3)
        # if params == 8:
        #     blur = cv2.medianBlur(img_data, 5)
        if params == 9:
            blur = cv2.blur(img_data, (6, 6))
        # if params == 9:
        #     blur = cv2.bilateralFilter(img_data, 6, 50, 50)
            # blur = cv2.bilateralFilter(img_data, 9, 75, 75)
        return blur

    def image_pixel_change(img_data, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img_data.shape
        img1d = np.ravel(img_data)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img

    def image_noise(img_data, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img_data.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img_data + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img_data)
            # Salt mode
            num_salt = np.ceil(amount * img_data.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img_data.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img_data.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img_data.shape]
            out[tuple(coords)]= 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img_data.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img_data + img_data * gauss
            return noisy.astype(np.uint8)

    '''    
    TODO: Add more mutators, current version is from DeepTest, https://arxiv.org/pdf/1708.08559.pdf

    Also check,   https://arxiv.org/pdf/1712.01785.pdf, and DeepExplore

    '''

    # TODO: Random L 0

    # TODO: Random L infinity

    # more transformations refer to: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html

    transformations = [image_contrast, image_brightness, image_blur, image_pixel_change, image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(map(lambda x: x * 0.1, list(range(5, 13)))))  # image_contrast
    params.append(list(range(0, 20)))  # image_brightness
    # params.append(list(range(1, 10)))  # image_blur
    params.append(list([1, 2, 3, 9]))
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 4)))  # image_noise
    # used = [0, 2, 3, 4]
    used = [0, 1, 2, 4]
    # classB = [5, 6]
    # classB = []
    @staticmethod
    def mutate(img_data, ref_img):
        random.seed(time.time())
        x, y, z = img_data.shape
        a = 0.05
        b = 0.5
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)
        trils = 100
        for i in range(trils):
            # Mu_here = Mutators()
            tid = random.sample(Mutators.used, 1)[0]
            transformation = Mutators.transformations[tid]

            params = Mutators.params[tid]
            param = random.sample(params, 1)[0]
            img_new = transformation(copy.deepcopy(img_data), param)

            sub = ref_img - img_new
            if np.max(abs(sub)) < l_infinity:
                return img_new
        return img_data
