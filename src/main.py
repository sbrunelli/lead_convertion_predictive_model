import numpy as np
from data_cleaning import DataCleaner
from features_engineering import FeatureExtractor
from model_selection import ModelSelector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.interactive(True)

if __name__ == '__main__':
    # read and clean the data
    dc = DataCleaner()
    data = dc.clean()

    # Debug transformations
    # data.to_csv('./data.csv', index=False, encoding='latin1')
    # assert False

    # separate target variable
    target = data.pop('Target')

    # train test split
    data_train, data_test, target_train, target_test = train_test_split(data, target)

    # featurize data
    featurizer = FeatureExtractor()
    X_train = featurizer.featurize(data_train)
    X_test = featurizer.featurize(data_test)

    # Convert to numpy arrays
    y_train = np.array(target_train)
    y_test = np.array(target_test)

    # Select model
    ms = ModelSelector()
    best_model = ms.get_best_model(X_train, X_test, y_train, y_test)

    # Print model scores
    print
    print
    print '   MODELS EVALUATION'
    print
    print '<Benchmark: Test dataset>'
    print ' + Positives: {:.5f}'.format(np.mean(y_test))
    print ' + Negatives: {:.5f}'.format(1 - np.mean(y_test))
    ms.print_model_scores(X_test)

    # # Evaluation plots, feature importances
    # topN = 20
    # classifiers = ms.get_all_classifiers()
    # feature_names = featurizer.get_dummified_features_names()
    #
    # # Logistic regression
    #
    # # Random forests
    # rf = classifiers[1]
    # rf_feature_importances = rf.feature_importances_
    # rf_feature_importances = rf_feature_importances / rf_feature_importances.max()
    # rf_feature_importances_idx_sorted = np.argsort(rf_feature_importances)
    # rf_bar_pos = np.arange(rf_feature_importances_idx_sorted.shape[0]) + .5
    # fig_feat_imp_rf = plt.figure(figsize=(12,9))
    # plt.barh(rf_bar_pos[-topN:], rf_feature_importances[rf_feature_importances_idx_sorted][-topN:], align='center', color='chocolate')
    # plt.yticks(rf_bar_pos[-topN:], feature_names[rf_feature_importances_idx_sorted][-topN:])
    # plt.xlabel('Feature importances')
    # plt.title('Random Forest', fontsize=14)
    # plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    # plt.show()
    #
    # # Gradient boosting
    # gb = classifiers[2]
    # gb_feature_importances = gb.feature_importances_
    # gb_feature_importances = gb_feature_importances / gb_feature_importances.max()
    # gb_feature_importances_idx_sorted = np.argsort(gb_feature_importances)
    # gb_bar_pos = np.arange(gb_feature_importances_idx_sorted.shape[0]) + .5
    # fig_feat_imp_gb = plt.figure(figsize=(12,9))
    # plt.barh(gb_bar_pos[-topN:], gb_feature_importances[gb_feature_importances_idx_sorted][-topN:], align='center', color='grey')
    # plt.yticks(gb_bar_pos[-topN:], feature_names[gb_feature_importances_idx_sorted][-topN:])
    # plt.xlabel('Feature importances')
    # plt.title('Random Forest', fontsize=14)
    # plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    # plt.show()
