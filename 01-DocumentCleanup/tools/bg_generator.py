import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_background():
    root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../dataset"
    noisy_folder = os.path.join(root_dir, 'test/noisy_data')
    output_folder = os.path.join(root_dir, 'background')

    print("Background Generator")
    print("input_data: ", noisy_folder)
    for file in tqdm(os.listdir(noisy_folder)):
        img = cv2.imread(os.path.join(noisy_folder, file), cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel)
        img = cv2.medianBlur(img, 9)
        cv2.imwrite(os.path.join(output_folder, file), img)
    print("Finished.")

if __name__ == "__main__":
    generate_background()