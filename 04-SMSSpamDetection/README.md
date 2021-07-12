# SMS Spam Detection
-----
## Problem description

The SMS Ham-Spam detection dataset is a set of SMS tagged messages that have been collected for SMS Spam research. It contains a set of 5,574 SMS messages in English, considering both train and test data. The tagging standard was defined as `ham` (legitimate) or `spam`.

## Objective

The goal of the this test is to achieve a model that can correctly manage the incoming messages on SMS format (`ham` or `spam`). Considering a real scenario, assume that a regular person does not want to see a `spam` message. However, they accepts if a normal message (`ham`) is sometimes allocated at the `spam` box.

## Proposed solution

The proposed solution consists of an Multinomial Naives Bayes classifier that classifies the input features as ham or spam. The feature vector was calculated using the Tf-Idf (Term frequency–Inverse document frequency) statistics which reflects the importance of the words in the document.

## Preparing environment

Create conda env:
```
$ conda create -n <env_name> python=3.7
$ conda activate <env_name>
```

Install the requirements:
```bash
$ pip install -r requirements.txt
```

## Running

Run the following command to perform sms spam detection for the test dataset.

```bash
$ python spam_detection.py
```

optional arguments:
| Arg | type | Description | Default |
| --- | ---- | ----------- | ------- |
| --model_path | str | Inference model path. | model/spam_detection_model.pkl |
| --vectorizer_path | str | Path of the TfidfVectorizer object used to transform messages to a doc-term matrix. | model/vectorizer.pkl |
| --input_csv | str | Input csv file path. | TestSet/sms-hamspam-test.csv |

## Training

If you would like to retrain the inference model, just execute the following command.

```bash
$ python tools/train.py
```
The new model will be saved in the */model* folder.