
import os
import sys
import cv2
import torch
import argparse
import numpy as np
import imageio as io
import matplotlib.pyplot as plt

from tqdm import tqdm
from tools.customdatasets import transform
from tools.post_processing import rectify_image
from net_models import ResNetUNet

class DocumentCleanup():
    def __init__(self, debug=False, net_weights='unet_best.pth'):
        self.debug = debug

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

        self.model = self.load_network(net_weights)

    def load_network(self, net_weights):
        # Instantiate network model
        net = ResNetUNet(pretrained=True, num_classes=1)

        # Load a pretrained network
        try:
            weights_folder = 'net_weights'
            weights_path = os.path.join(weights_folder, net_weights)

            if not os.path.exists(weights_folder):
                os.makedirs(weights_folder)

            if not os.path.exists(weights_path):
                raise Exception("You need a network weights file.")

            net.load_state_dict(torch.load(weights_path))

        except Exception as err:
            print("A problem has occurred loading model file.", err)
        
        return net.to(self.device).eval()

    def post_processing(self, image):
        # Rectify and binarize network output image.
        image = rectify_image(image)
        return image

    def load_image(self, file_path):
        
        image = io.imread(file_path)
        image = transform(image)
        return image

    def process(self, input_path, output_path):
        with torch.no_grad():
            image = self.load_image(input_path)

            if len(image.shape) < 4:
                image = image.unsqueeze_(0)

            image = image.to(self.device)

            output = self.model(image)[0]
            output = output.cpu().squeeze()
            output = self.post_processing(output.numpy())

            if self.debug:
                f, axarr = plt.subplots(1, 2)
                axarr[0].imshow(image.cpu()[0][0], cmap='gray', vmin=0, vmax=1)
                axarr[1].imshow(output, cmap='gray', vmin=0, vmax=255)
                plt.show()
            else:
                cv2.imwrite(output_path, output)


def create_parser_arguments():
    parser = argparse.ArgumentParser(description="Document Cleanup")
    parser.add_argument(
        "--input_folder",
        dest="input_folder",
        help="Folder containing images to be cleaned.",
        default="dataset/test/noisy_data"
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        help="Folder where the processed imagens will be saved.",
        default="dataset/test/clean_data"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        help="Run in debug mode. [y/n]",
        default="n"
    )
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    return args


if __name__=="__main__":
    args = create_parser_arguments()
    debug = args.debug.lower() == 'y'

    doc = DocumentCleanup(debug)
    for file in tqdm(os.listdir(args.input_folder)):
        input_path = os.path.join(args.input_folder, file)
        output_path = os.path.join(args.output_folder, file)
        doc.process(input_path, output_path)