import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, train_file_path):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english'
        )

        # Loading data
        df = open_sms_file(train_file_path)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            df['message'], df['label'], test_size = 0.2
        )

        self.x_train_vec = self.vectorize_data(self.x_train, fit=True)
        self.x_val_vec = self.vectorize_data(self.x_val)
        self.x_test_vec = None

    def vectorize_data(self, x, fit=False):
        if fit:
            return self.vectorizer.fit_transform(x)
        else:
            return self.vectorizer.transform(x)

    def save_vectorizer(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.vectorizer, fp)

    def get_train_data(self):
        return self.x_train_vec, self.y_train

    def get_val_data(self):
        return self.x_val_vec, self.y_val


def open_sms_file(path, test=False):
    """Load and preprocessing sms messages

    Args:
        path ([type]): Path of the file to be loaded.
        test (bool, optional): Wheter is loading a test file or not. Defaults to False.

    Returns:
        [pd.DataFrame]: Dataframe file with a list of labels and messages.
    """
    if not test:
        with open(path, 'r') as fp:
            lines = fp.readlines()
            lines = [ line.replace('\n', '').split('\t') for line in lines ]
    
        return pd.DataFrame([ {
                'label': 0 if line[0] == 'ham' else 1,
                'message': re.sub('[^a-zA-Z0-9 \n\.]', '', line[1])
            } for line in lines ])
    else:
        with open(path, 'r') as fp:
            lines = fp.readlines()
            lines = [ line.replace('\n', '') for line in lines ]

        return pd.DataFrame([ {
                'label': -1,
                'message': re.sub('[^a-zA-Z0-9 \n\.]', '', line)
            } for line in lines ])