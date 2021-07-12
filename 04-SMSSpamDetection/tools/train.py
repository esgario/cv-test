import os
import sys
import inspect
import pickle
from utils import Dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

current_path = os.path.dirname(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename))
parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

def train_model(train_file_path, save=False):
    """Training and saving the spam classifier model.

    Args:
        train_file_path ([str]): [Path of the training set csv file]
        save (bool, optional): [If true a new model will be saved]. Defaults to False.
    """
    # Loading data
    dt = Dataset(train_file_path)
    x_train, y_train = dt.get_train_data()
    x_val, y_val = dt.get_val_data()

    # Fitting model
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    # Save model file to be used in future inferences
    if save:
        file_path = os.path.join(parent_path, 'model/spam_detection_model.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(classifier, fp)
        dt.save_vectorizer(os.path.join(parent_path, 'model/vectorizer.pkl'))
        print("New model saved.")

    # Testing on validation subset
    predicted = classifier.predict(x_val)
    actual = y_val.tolist()

    # Printing results
    print('Accuracy: %.3f' % accuracy_score(actual, predicted))
    print('F-Measure: %.3f' % f1_score(actual, predicted, average='binary'))
    print('Confusion Matrix:')
    print(confusion_matrix(actual, predicted))        
    print('Report:', classification_report(actual, predicted))
    
if __name__ == "__main__":
    train_model(
        train_file_path=os.path.join(parent_path, 'TrainingSet/sms-hamspam-train.csv'),
        save=True
    )