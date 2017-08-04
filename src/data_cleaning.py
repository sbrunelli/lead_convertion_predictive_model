import pandas as pd
import numpy as np
import re
import datetime

class DataCleaner(object):

    def __init__(self):
        self.data = None

    def _read_data(self):
        self.data = pd.read_csv('./data/opportunities.csv', parse_dates=['Create.Day'], encoding='latin1', index_col='Opportunity.Number')

    def _clean_order_entry_chf(self):
        '''
        Cleans the Order.Entry.CHF variable.
        - Removes thousands ',' separator character
        - Drops all rows with z score for CHF greater than 10
        - Drops all rows with CHF value less than 1000
        '''

        self.data['Order.Entry.CHF'] = self.data['Order.Entry.CHF'].map(lambda x: float(re.sub(',','',x)))
        CHF_mean = self.data['Order.Entry.CHF'].mean()
        CHF_std = self.data['Order.Entry.CHF'].std()
        CHF_z = self.data['Order.Entry.CHF'].map(lambda x: (x-CHF_mean)/CHF_std)
        self.data = self.data[abs(CHF_z) <= 10]
        self.data = self.data[self.data['Order.Entry.CHF'] >= 1000]

    def _attach_one(self):
        '''
        Attaches a dummy column of ones to allow easy aggregation calculations
        '''
        self.data['_one'] = 1

    def _create_yesterday(self):
        '''
        Take the column Create.Day and creates a copy of it shifted back in time by 1 day
        '''
        self.data['Create.Yesterday'] = self.data['Create.Day'].map(lambda x: x-datetime.timedelta(days=1))

    def _create_target_variable(self):
        self.data['Target'] = self.data['Status.Category'].map(lambda x: 1 if x=='Won' else 0)

    def _drop_quantity_gt_one(self):
        self.data = self.data[self.data['Quantity'] == 1]

    def _drop_na_customers(self):
        '''
        Drops all rows with NA value for customer
        '''
        self.data = self.data[self.data['Customer'].notnull()]

    def _create_time_units(self):
        self.data['Year'] = self.data['Create.Day'].map(lambda x: x.year)
        self.data['Month'] = self.data['Create.Day'].map(lambda x: x.month)

    def _create_rolling_period_metrics(self, window_size, number_periods, period_suffix, group_dimension):
        '''
        This functions creates rolling metrics for:
            - Number of Contacts
            - Number of Wins
            - Convertion Ratio

        against a given dimension and for a given window_size.
        '''

        # create timedelta in days
        if window_size=='month':
            multiplier=30
        elif window_size=='year':
            multiplier=365
        timedelta = multiplier * number_periods

        # sort data by Customer, Create.Day
        self.data.sort_values(by=['Customer', 'Create.Day'], inplace=True)

        # create rolling series
        rolling_metric = self.data[[group_dimension,
                             'Create.Yesterday',
                             'Target',
                            '_one']].groupby(group_dimension).rolling(window=datetime.timedelta(days=timedelta),
                                                                          on='Create.Yesterday',
                                                                          min_periods=1).sum()[['Target', '_one']]

        # convert series to data frame
        rolling_metric = pd.DataFrame(rolling_metric)

        # Reset index on temporary data frame
        rolling_metric.reset_index(inplace=True)

        # Rename columns
        n_wins_metric_name = '_'+group_dimension+'.Win.'+period_suffix
        n_contacts_metric_name = '_'+group_dimension+'.Contacts.'+period_suffix
        rolling_metric.rename(columns={'Target': n_wins_metric_name,
                                       '_one': n_contacts_metric_name},
                              inplace=True)

        # Drop customer columns
        rolling_metric.drop('Customer', axis=1, inplace=True)

        # Reset index on main data frame
        self.data.reset_index(inplace=True)

        # Merge df with rolling_metric
        self.data = self.data.merge(rolling_metric, on='Opportunity.Number', how='left')

        # Substract same day's victories:
        # everything that happened the same day as current opportunity has no influence on the previous relationship
        # between Trivadis and the customer.
        # Current outcome is leaky, must be discarded too.
        self.data.set_index('Opportunity.Number', inplace=True)
        rolling_metric_yesterday = pd.DataFrame(self.data[[group_dimension,
                                                    'Create.Yesterday',
                                                    'Target',
                                                     '_one']].groupby(['Customer',
                                                                         'Create.Yesterday']).cumsum()[['Target',
                                                                                                      '_one']]).reset_index()
        rolling_metric_yesterday.rename(columns={'Target': n_wins_metric_name+'_yesterday',
                                                '_one': n_contacts_metric_name+'_yesterday'},
                                        inplace=True)
        self.data.reset_index(inplace=True)
        self.data = self.data.merge(rolling_metric_yesterday, on='Opportunity.Number', how='left')

        self.data[n_contacts_metric_name[1:]] = self.data[n_contacts_metric_name] - self.data[n_contacts_metric_name+'_yesterday']
        self.data[n_wins_metric_name[1:]] = self.data[n_wins_metric_name] - self.data[n_wins_metric_name+'_yesterday']
        self.data[group_dimension+'.ConvRatio.'+period_suffix] = self.data[n_wins_metric_name[1:]] / self.data[n_contacts_metric_name[1:]]
        self.data[group_dimension+'.ConvRatio.'+period_suffix] = self.data[group_dimension+'.ConvRatio.'+period_suffix].fillna(.0)

        # Reset data frame index to Opportunity Number
        self.data.set_index('Opportunity.Number', inplace=True)

    def _create_trends(self):

        # Selects subset of columns to use for trends calculations
        columns = ['Customer.ConvRatio.LY',
                  'Customer.ConvRatio.LS',
                  'Customer.ConvRatio.LQ',
                  'Customer.ConvRatio.LM']
        df_matrix = self.data[columns].copy().values

        # Creates initial trends array
        nrows = df_matrix.shape[0]
        trends = np.chararray(shape=nrows, itemsize=100)
        trends[:] = 'None'

        # Creates trends masks
        long_term_down_mask = ((df_matrix[:,0] > df_matrix[:,1]) & (df_matrix[:,1] > df_matrix[:,2]) & (df_matrix[:,2] > df_matrix[:,3]))
        long_term_up_mask = ((df_matrix[:,0] < df_matrix[:,1]) & (df_matrix[:,1] < df_matrix[:,2]) & (df_matrix[:,2] < df_matrix[:,3]))
        mid_term_down_mask = ((df_matrix[:,1] > df_matrix[:,2]) & (df_matrix[:,2] > df_matrix[:,3]))
        mid_term_up_mask = ((df_matrix[:,1] < df_matrix[:,2]) & (df_matrix[:,2] < df_matrix[:,3]))
        short_term_down_mask = ((df_matrix[:,2] > df_matrix[:,3]))
        short_term_up_mask = ((df_matrix[:,2] < df_matrix[:,3]))

        # Applies masks, short period to long period priority (long period trends are supposed to be stronger)
        trends[short_term_up_mask] = 'Short term growth'
        trends[short_term_down_mask] = 'Short term decrease'
        trends[mid_term_up_mask] = 'Mid term growth'
        trends[mid_term_down_mask] = 'Mid term decrease'
        trends[long_term_up_mask] = 'Long term growth'
        trends[long_term_down_mask] = 'Long term decrease'

        # Appends trends as new columns to data frame
        self.data['Customer.Trends'] = trends

    def _drop_temporary_columns(self):
        '''
        Drops all remaining temporary columns, those that start with _
        '''
        def drop_column(col):
            del self.data[col]

        [drop_column(column) for column in self.data.columns.tolist() if column.startswith('_')]

    def _drop_initial_bulk_load(self):
        '''
        Drops all samples loaded on the 03/01/2009 as they represent an initial load, probably an import from a previous system and contain no date information, thus potentially messing up the training process that strongly depends upon that logic
        '''
        mask = self.data['Create.Day'] == '2009-03-01'
        self.data = self.data[~(mask)]

    def clean(self):
        self._read_data()
        self._drop_quantity_gt_one()
        self._drop_na_customers()
        self._clean_order_entry_chf()
        self._attach_one()
        self._create_yesterday()
        self._create_time_units()
        self._create_target_variable()
        self._create_rolling_period_metrics(window_size='year', number_periods=1, period_suffix='LY', group_dimension='Customer')
        self._create_rolling_period_metrics(window_size='month', number_periods=6, period_suffix='LS', group_dimension='Customer')
        self._create_rolling_period_metrics(window_size='month', number_periods=3, period_suffix='LQ', group_dimension='Customer')
        self._create_rolling_period_metrics(window_size='month', number_periods=1, period_suffix='LM', group_dimension='Customer')
        self._create_trends()
        self._drop_temporary_columns()
        # self._drop_initial_bulk_load()
        return self.data
