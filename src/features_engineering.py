import pandas as pd
import numpy as np

class FeatureExtractor(object):

    def __init__(self):
        self.data = None
        self._features_names = None
        self._dummified_features_names = None

    def _subset_columns(self):
        columns = [
        'Order.Entry.CHF',
        'Year',
        'Month',
        'Customer.Contacts.LY',
        'Customer.Win.LY',
        'Customer.ConvRatio.LY',
        'Customer.Contacts.LS',
        'Customer.Win.LS',
        'Customer.ConvRatio.LS',
        'Customer.Contacts.LQ',
        'Customer.Win.LQ',
        'Customer.ConvRatio.LQ',
        'Customer.Contacts.LM',
        'Customer.Win.LM',
        'Customer.ConvRatio.LM',
        'Customer.Trends'
        # 'Customer.Industry',
        # 'Customer.First.Contact',
        # 'Customer.Contacts.So.Far',
        # 'Customer.ConvRatio.So.Far',
        # 'Customer.ConvRatio.Last5',
        # 'Customer.ConvRatio.LastYear',
        # 'Customer.Order.Entry.CHF.std2avg',
        # 'Customer.Last.Target',
        # 'Customer.Days.Since.LastContact',
        # 'Customer.Days.Since.LastWin',
        # 'Customer.CHF.Last.Won',
        # 'Customer.Days.Since.LastLoss',
        # 'Customer.CHF.Last.Loss'#,
        ]
        self.data = self.data[columns]

    def _dummify(self):
        self._features_names = self.data.columns.values
        self.data = pd.get_dummies(self.data, drop_first=True)
        self._dummified_features_names = self.data.columns.values

    def get_features_names(self):
        '''
        Returns features names before dummification of categorical variables
        '''
        return self._features_names

    def get_dummified_features_names(self):
        '''
        Returns features names after dummification of categorical variables
        '''
        return self._dummified_features_names

    def _create_missing_flags(self):
        '''
        For each column that contains NAs, before imputing those values, a flag is created that reports whether the 0 value was a missing value or a real 0 originally
        '''
        null_columns = self.data.columns[(self.data.apply(lambda x: np.sum(x.isnull())) > 0).values]
        self._dummified_features_names = self._dummified_features_names.tolist()
        for nc in null_columns:
            new_col_name = nc + '.missing'
            self._dummified_features_names.append(new_col_name)
            self.data[new_col_name] = self.data[nc]
            self.data[new_col_name] = self.data[new_col_name].map(lambda x: 1 if np.isnan(x) else 0)
        self._dummified_features_names = np.array(self._dummified_features_names)

    def _impute_nas_as_zeros(self):
        '''
        Fills all NA values with 0 for all features
        '''
        self.data = self.data.fillna(0)

    def featurize(self, data):
        self.data = data
        self._subset_columns()
        self._dummify()
        self._create_missing_flags()
        self._impute_nas_as_zeros()
        return self.data.values
