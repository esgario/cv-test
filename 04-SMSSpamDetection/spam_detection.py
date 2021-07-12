import os
import sys
import pickle
import argparse
import numpy as np
from tools.utils import open_sms_file

class SpamDetector():
    def __init__(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as fp:
            self.model = pickle.load(fp)
        
        with open(vectorizer_path, 'rb') as fp:
            self.vectorizer = pickle.load(fp)

    def save_results(self, output_filename, df, predicted):
        print(f"Saving results to: {output_filename}")
        df['label'] = [ 'ham' if x==0 else 'spam' for x in predicted ]
        df.to_csv(output_filename, index=False)

    def inference(self, csv_test_file):
        print("Starting inference")

        # Open csv file
        df = open_sms_file(csv_test_file, test=True)

        # Convert messages to numerical features
        x_test = self.vectorizer.transform(df['message'].tolist())

        # Do inference
        predicted = self.model.predict(x_test)

        n_ham = np.sum(np.array(predicted) == 0)
        n_spam = np.sum(np.array(predicted) == 1)
        print("Detections:\n# ham: %d\n# spam: %d" % (n_ham, n_spam))

        # Save final results
        path = os.path.dirname(csv_test_file)
        filename = os.path.basename(csv_test_file).split('.')
        filename = filename[0] + "_results." + filename[1]
        self.save_results(os.path.join(path, filename), df, predicted)

        print("Done!")


def create_parser_arguments():
    parser = argparse.ArgumentParser(description="Document Cleanup")

    parser.add_argument(
        "--model_path",
        dest="model_path",
        help="Inference model path.",
        default="model/spam_detection_model.pkl"
    )
    parser.add_argument(
        "--vectorizer_path",
        dest="vectorizer_path",
        help="Path of the TfidfVectorizer object used to transform messages to a doc-term matrix.",
        default="model/vectorizer.pkl"
    )
    parser.add_argument(
        "--input_csv",
        dest="input_csv",
        help="Input csv file path.",
        default="TestSet/sms-hamspam-test.csv"
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
    return args


if __name__ == "__main__":
    args = create_parser_arguments()
    spam_detector = SpamDetector(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path
    )
    spam_detector.inference(args.input_csv)