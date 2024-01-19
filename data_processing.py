import pandas as pd
from sklearn import preprocessing

"""
Preprocessing Dataset
"""

def preprocess_data(data):
    data = data.drop(["Unnamed: 0"], axis=1)
    # ... other preprocessing steps
    return data

def encode_labels(data):
    le = preprocessing.LabelEncoder()
    le.fit(data['label'])
    data['label'] = le.transform(data['label'])

    return data