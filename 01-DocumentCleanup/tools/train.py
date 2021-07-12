import os
import sys
import inspect
import warnings
import argparse
from classifier import Classifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Application responsible for detecting lens flare in images.')

    parser.add_argument('--optimizer', type=str, default='sgd',
                    help='[TRAIN] Select the optimizer method (sgd/adam)')
    parser.add_argument('--batch_size', type=int, default=4,
                    help='[TRAIN] Number of training examples utilized in one iteration.')
    parser.add_argument('--num_classes', type=int, default=1,
                    help='[TRAIN] Number of model classes.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='[TRAIN] Weight decay (L2 penaulty)')
    parser.add_argument('--model', type=str, default='unet',
                    help='[TRAIN] CNN model that will be used in training.')
    parser.add_argument('--epochs', type=int, default=80,
                    help='[TRAIN] Number of epochs.')
    parser.add_argument('--pretrained', type=bool, default=True,
                    help='[TRAIN] Perform training using a pretrained model?')
    parser.add_argument('--filename', type=str, default='unet',
                    help='[TRAIN] Name of the output model file.')
    parser.add_argument('--images_dir', type=str, default='dataset',
                    help='[TRAIN] Directory of the dataset.')
    parser.add_argument('--use_snapshot', action='store_true',
                    help='[TRAIN] Continue training from the last snapshot.')
    parser.add_argument('--test', action='store_true',
                    help='[TEST] Flag to enable test mode.')

    options = parser.parse_args()

    classifier = Classifier(parser)

    if options.test:
        classifier.run_test()
    else:
        classifier.run_training()
        
