import cv2
import math
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


# ----------------------------- STANDARD

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = True

    def __call__(self, data, target):
        if isinstance(data, np.ndarray):
            data = Image.fromarray(data)
            target = Image.fromarray(target)
            
            self.PIL2Numpy = True

        for a in self.augmentations:
            data, target = a(data, target)

        if self.PIL2Numpy:
            data, target = np.array(data), np.array(target)

        return data, target



class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, data, target):
        return (
            tf.adjust_gamma(data, random.uniform(1, 1 + self.gamma)),
            target
        )


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, data, target):
        return (
            tf.adjust_saturation(data, random.uniform(1 - self.saturation, 1 + self.saturation)),
            target,
        )


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, data, target):
        return (
            tf.adjust_hue(data, random.uniform(-self.hue, self.hue)),
            target,
        )


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, data, target):
        return (
            tf.adjust_brightness(data, random.uniform(1 - self.bf, 1 + self.bf)),
            target,
        )


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, data, target):
        return (
            tf.adjust_contrast(data, random.uniform(1 - self.cf, 1 + self.cf)),
            target,
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data, target):
        if random.random() < self.p:
            return (data.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT))
        return data, target


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data, target):
        if random.random() < self.p:
            return (data.transpose(Image.FLIP_TOP_BOTTOM), target.transpose(Image.FLIP_TOP_BOTTOM))
        return data, target


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, data, target):
        return (data.resize(self.size, Image.BILINEAR), target.resize(self.size, Image.NEAREST))


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data, target):
        w, h = data.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return data, target
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (data.resize((ow, oh), Image.BILINEAR), target.resize((ow, oh), Image.BILINEAR))
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (data.resize((ow, oh), Image.BILINEAR), target.resize((ow, oh), Image.BILINEAR))
