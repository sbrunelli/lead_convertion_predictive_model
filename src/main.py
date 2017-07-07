import numpy as np
from data_cleaning import DataCleaner
from features_engineering import FeatureExtractor
from model_selection import ModelSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # read and clean the data
    dc = DataCleaner()
    data = dc.clean()

    # separate target variable
    target = data.pop('target')

    # train test split
    data_train, data_test, target_train, target_test = train_test_split(data, target)

    # featurize data
    featurizer = FeatureExtractor()
    X_train = featurizer.featurize(data_train)
    X_test = featurizer.featurize(data_test)

    # Convert to numpy arrays
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_train = np.array(target_train)
    y_test = np.array(target_test)

    # Select model
    ms = ModelSelector()
    best_model = ms.get_best_model(X_train, X_test, y_train, y_test)

    # Print model scores
    ms.print_model_scores()
