import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def reescale_image(image, height=None, width=None):
    if height:
        if image.shape[0] != height:
            hpercent = (height/float(image.shape[0]))
            width = int((float(image.shape[1])*float(hpercent)))
            resized_image = cv2.resize(image, (width, height))
    elif width:
        if image.shape[1] != width:
            wpercent = (width/float(image.shape[1]))
            height = int((float(image.shape[0])*float(wpercent)))
            resized_image = cv2.resize(image, (width, height))
    else:
        resized_image = image

    return resized_image


def rotate_image(image, angle_min, angle_max):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    angle = random.randrange(angle_min, angle_max, 1)

    height, width = image.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), borderValue=(1,1,1))
    return rotated_mat

def shearing_image(image, p=0.1):
    srcTri = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [0, image.shape[0] - 1]
    ]).astype(np.float32)

    r = [ random.randrange(0, int(p * 100), 1) / 100 for i in range(6) ]
    dstTri = np.array([
        [image.shape[1] * r[0], image.shape[0] * r[1]],
        [image.shape[1] * (1 - r[2])-20, image.shape[0] * r[3]],
        [image.shape[1] * r[4], image.shape[0] * (1 - r[5])]
    ]).astype(np.float32)

    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    image = cv2.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]), borderValue=(1,1,1))

    return image

def bg_transforms(image):
    if random.random() > 0.5:
        # flip vertically
        image = cv2.flip(image, 0)

    if random.random() > 0.5:
        # flip horizontally
        image = cv2.flip(image, 1)

    if random.random() > 0.5:
        p = random.random()
        image = image ** (1+p)

    return image

def text_transforms(image):
    image = shearing_image(image, p=0.1)
    image = rotate_image(image, angle_min=-10, angle_max=10)

    return image

def generate_image(bg_path, text_path, output_width=640):
    bg_image = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    text_image = cv2.imread(text_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    bg_image = bg_transforms(bg_image)
    text_image = text_transforms(text_image)

    text_image = reescale_image(text_image, width=output_width)
    bg_image = cv2.resize(bg_image, (text_image.shape[1], text_image.shape[0]))

    s = random.randrange(0, 20, 1) / 100
    noisy_image = bg_image * (text_image + s)
    
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    noisy_image = (noisy_image * 255).astype(np.uint8)
    text_image = (text_image * 255).astype(np.uint8)

    return noisy_image, text_image


if __name__ == "__main__":
    data_folder = os.path.dirname(os.path.realpath(__file__)) + "/../dataset"
    bg_folder = "background"
    text_folder = "plain_text"
    
    bg_files = os.listdir(os.path.join(data_folder, bg_folder))
    text_files = os.listdir(os.path.join(data_folder, text_folder))

    n_images = [ 200, 50 ]
    output_folder = [ 'train', 'val' ]

    for n, folder in zip(n_images, output_folder):
        print(f"Generating {folder.upper()} Dataset.")
        for i in tqdm(range(n)):
            try:
                bg_path = os.path.join(data_folder, bg_folder, random.choice(bg_files))
                text_path = os.path.join(data_folder, text_folder, random.choice(text_files))

                noisy_image, clean_image = generate_image(bg_path, text_path)
                output_name = str(i).zfill(3) + '.png'
                cv2.imwrite(os.path.join(data_folder, folder, 'noisy_data', output_name), noisy_image)
                cv2.imwrite(os.path.join(data_folder, folder, 'clean_data', output_name), clean_image)
            except Exception as err:
                print(f"Failed to generate image. bg_path: {bg_path}, text_path: {text_path}. err: {err}")

    print("Finished")