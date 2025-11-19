from __future__ import absolute_import

import torch
#from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class RandomOcc(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, sl=0.02, sh=0.05, r1=0.1, mean=[0.485, 0.456, 0.406]):


        self.sl = sl
        self.sh = sh
        self.r1 = r1

        self.mean = mean


    def __call__(self, img):
        # theta = random.uniform(self.sl, self.sh)
        # mask = np.random.uniform(0,1,(1,img.size()[1],img.size()[2])) < theta
        # mask = torch.from_numpy(mask).expand_as(img)
        # mean = torch.tensor(self.mean).view(3,1,1).expand_as(img)
        # occ = mask*img+(~mask)*mean

        # occ = torch.tensor(self.mean).view(3,1,1).expand_as(img)
        occ = torch.zeros_like(img)
        for attempt in range(100):

            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                     occ[0, x1:x1 + h, y1:y1 + w] = img[0, x1:x1 + h, y1:y1 + w]
                     occ[1, x1:x1 + h, y1:y1 + w] = img[1, x1:x1 + h, y1:y1 + w]
                     occ[2, x1:x1 + h, y1:y1 + w] = img[2, x1:x1 + h, y1:y1 + w]
                else:
                     occ[0, x1:x1 + h, y1:y1 + w] = img[0, x1:x1 + h, y1:y1 + w]
                return occ



        return occ





class RandomShuffleChannels(object):

    def __init__(self, probability: float = 0.2):
        self.probability = probability


    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        img_array = np.array(img)
        channels = [0,1,2]
        random.shuffle(channels)
        shuffled_array = img_array[:,:,channels]
        shuffled_img = Image.fromarray(shuffled_array)

        return shuffled_img


class RandomSelectChannel(object):

    def __init__(self, probability: float = 0.2):
        self.probability = probability


    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img




        if random.uniform(0, 1) > self.probability: # rgb 随机选一个通道
            img_array = np.array(img)
            channel_index = random.choice([0, 1, 2])
            select_channel = img_array[:, :, channel_index]
            replicated_img = np.stack([select_channel, select_channel, select_channel], axis=2)
            replicated_img = Image.fromarray(replicated_img)
            return replicated_img
        else:#随即灰度化
            img_gray = np.array(img.convert('L'))
            replicated_img = np.stack([img_gray, img_gray, img_gray], axis=2)
            replicated_img = Image.fromarray(replicated_img)
            return replicated_img













class RandomGrayscaleErasing(object):
    """ Randomly selects a rectangle region in an image and use grayscale image
        instead of its pixels.
        'Local Grayscale Transfomation' by Yunpeng Gong.
        See https://arxiv.org/pdf/2101.08533.pdf
    Args:
         probability: The probability that the Random Grayscale Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
    """

    def __init__(self, probability: float = 0.2, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """
        Args:
            img: after ToTensor() and Normalize([...]), img's type is Tensor
        """
        if random.uniform(0, 1) > self.probability:
            return img

        height, width = img.size()[-2], img.size()[-1]
        area = height * width

        for _ in range(100):

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)  # height / width

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                # tl
                x = random.randint(0, height - h)
                y = random.randint(0, width - w)
                # unbind channel dim
                r, g, b = img.unbind(dim=-3)
                # Weighted average method -> grayscale patch
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
                l_img = l_img.unsqueeze(dim=-3)  # rebind channel
                # erasing
                img[0, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[1, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]
                img[2, y:y + h, x:x + w] = l_img[0, y:y + h, x:x + w]

                return img

        return img
