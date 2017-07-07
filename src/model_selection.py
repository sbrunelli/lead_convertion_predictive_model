import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelSelector(object):

    def __init__(self):
        self.classifiers = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]
        self.scores = []
        self.best_model_index = None

    def _train_models(self, X_train, y_train):
        for classifier in self.classifiers:
            classifier.fit(X_train, y_train)

    def _score_models(self, X_test, y_test):
        for classifier in self.classifiers:
            y_predicted = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_predicted)
            precision = precision_score(y_test, y_predicted)
            recall = recall_score(y_test, y_predicted)
            f1 = f1_score(y_test, y_predicted)
            self.scores.append((accuracy, precision, recall, f1))

    def _select_best_model(self, metric='accuracy'):
        metric_index_map = {'accuracy': 0, 'precision': 1, 'recall': 2, 'f1': 3}
        scores_matrix = np.array(self.scores)
        optimizing_metric = scores_matrix[:, metric_index_map.get(metric)]
        self.best_model_index = np.argmax(optimizing_metric)

    def print_model_scores(self):
        for idx, classifier in enumerate(self.classifiers):
            print classifier.__class__
            print ' #  Accuracy: {:.5f}'.format(self.scores[idx][0])
            print ' # Precision: {:.5f}'.format(self.scores[idx][1])
            print ' #    Recall: {:.5f}'.format(self.scores[idx][2])
            print ' #  F1 Score: {:.5f}'.format(self.scores[idx][3])
            #print ' # {:s} Accuracy: {:.5f}'.format(classifier.__class__, score)

    def get_best_model(self, X_train, X_test, y_train, y_test):
        self._train_models(X_train, y_train)
        self._score_models(X_test, y_test)
        self._select_best_model()
        return self.classifiers[self.best_model_index]
