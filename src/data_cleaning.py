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
        self.data['one'] = 1

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

    def _create_customer_metrics(self):
        '''
        Creates all features related to relationship between Trivadis and the customers. These include:

        * 1st contact?
        * how many contacts so far?
        * convertion rate so far
        * how many standard deviations is current offer removed from average
        * convertion rate over last 5 opportunities
        * how many contracts during last rolling year
        * convertion rate during last rolling year
        * last outcome
        * time (days) since last offer
        * time (days) since last Won
        * time (days) since last loss
        * Amount CHF last Won
        * Amount CHF last loss
        '''

        def first_contact():
            self.data = self.data.reset_index()
            self.data = self.data.sort_values(by=['Customer', 'Create.Day', 'Opportunity.Number'])
            self.data['Customer.Contacts.So.Far'] = self.data.groupby('Customer').cumsum()['one']
            self.data['Customer.Contacts.So.Far'] = self.data['Customer.Contacts.So.Far'].map(lambda x: x - 1)
            self.data['Customer.First.Contact'] = self.data['Customer.Contacts.So.Far'].map(lambda x: 1 if x==0 else 0)
            self.data.set_index('Opportunity.Number', inplace=True)

        # def conv_rate_so_far():
        #     self.data['Customer.Won.So.Far'] = self.data.groupby('Customer').cumsum()['Target']
        #     self.data['Customer.ConvRatio.So.Far'] = self.data['Customer.Won.So.Far'].astype('float') / self.data['Customer.Contacts.So.Far']
        #     # Shift it by 1 to avoid leakage
        #     self.data['Customer.ConvRatio.So.Far'] = self.data.groupby('Customer')['Customer.ConvRatio.So.Far'].shift(1)

        def conv_rate_so_far():
            self.data['Customer.Won.So.Far'] = self.data.groupby('Customer').cumsum()['Target']
            # Shift it by 1 to avoid leakage
            self.data['Customer.Won.So.Far'] = self.data.groupby('Customer')['Customer.Won.So.Far'].shift(1)
            self.data['Customer.ConvRatio.So.Far'] = self.data['Customer.Won.So.Far'].astype('float') / (self.data['Customer.Contacts.So.Far'])

        def conv_rate_last_five():
            '''
            Adds the features:
                - Customer.Won.Last5
                - Customer.ConvRatio.Last5
            '''
            # cust = 'Audi AG [2101]'
            cvrl5 = self.data.groupby('Customer').rolling(window=5, on='Create.Day', min_periods=1).sum()['Target']
            cvrl5 = cvrl5.reset_index()
            cvrl5.drop('Customer', axis=1, inplace=True)
            cvrl5.rename(columns={'Target': 'Customer.Won.Last5', 'level_1': 'Opportunity.Number'}, inplace=True)
            self.data = self.data.reset_index()
            self.data = self.data.merge(cvrl5, on='Opportunity.Number', how='left')
            # Shift it by 1 to avoid leakage
            self.data['Customer.Won.Last5'] = self.data.groupby('Customer')['Customer.Won.Last5'].shift(1)
            self.data['Customer.ConvRatio.Last5'] = self.data['Customer.Won.Last5'] / self.data['Customer.Contacts.So.Far'].map(lambda x: min(x, 5))
            self.data.set_index('Opportunity.Number', inplace=True)

        def conv_rate_last_year():
            cvrly = self.data.groupby('Customer', ).rolling(window=datetime.timedelta(days=365), on='Create.Day', min_periods=1).sum()[['Target', 'one']]
            cvrly = cvrly.reset_index()
            print cvrly.head(5)
            cvrly.drop('Customer', axis=1, inplace=True)
            cvrly.rename(columns={'Target': 'Customer.Won.LastYear', 'one': 'Customer.Contacts.LastYear', 'level_1': 'Opportunity.Number'}, inplace=True)
            self.data = self.data.reset_index()
            self.data = self.data.merge(cvrly, on='Opportunity.Number', how='left')
            self.data['Customer.Contacts.LastYear'] = self.data.groupby('Customer')['Customer.Contacts.LastYear'].shift(1)
            # Shift it by 1 to avoid leakage
            self.data['Customer.Won.LastYear'] = self.data.groupby('Customer')['Customer.Won.LastYear'].shift(1)
            self.data['Customer.ConvRatio.LastYear'] = self.data['Customer.Won.LastYear'] / self.data['Customer.Contacts.LastYear']
            self.data = self.data.set_index('Opportunity.Number')

        def stddev_offer_to_avg_contract_size():
            self.data['Order.Entry.CHF.Won'] = self.data['Order.Entry.CHF'] * self.data['Target']
            # Shift it by 1 to avoid leakage
            self.data['Customer.Avg.Order.Entry.CHF.So.Far'] = self.data.groupby('Customer').cumsum()['Order.Entry.CHF.Won'] / self.data['Customer.Won.So.Far']
            self.data['Customer.Avg.Order.Entry.CHF.So.Far.Lag1'] = self.data.groupby('Customer')['Customer.Avg.Order.Entry.CHF.So.Far'].shift(1)
            self.data['Customer.Order.Entry.CHF.std2avg'] = (self.data['Order.Entry.CHF'] - self.data['Customer.Avg.Order.Entry.CHF.So.Far.Lag1']) /  \
                    self.data['Customer.Avg.Order.Entry.CHF.So.Far.Lag1'].map(lambda x: np.nan if x==0 else x)
            self.data['Customer.Order.Entry.CHF.std2avg'] = self.data['Customer.Order.Entry.CHF.std2avg'].fillna(0)
            self.data.drop(['Customer.Avg.Order.Entry.CHF.So.Far', 'Customer.Avg.Order.Entry.CHF.So.Far.Lag1'], axis=1, inplace=True)

        def last_outcome():
            self.data['Customer.Last.Target'] = self.data.groupby('Customer')['Target'].shift(1)

        def time_since_last_contact():
            self.data['Customer.Days.Since.LastContact'] = (self.data['Create.Day'] - self.data.groupby('Customer')['Create.Day'].shift(1)).dt.days

        def time_since_chf_last_win():
            idxWon = (self.data.Target == 1)
            self.data.loc[idxWon, 'Customer.Last.Day.Won'] = self.data.loc[idxWon, 'Create.Day']
            self.data['Customer.Last.Day.Won'] = self.data.groupby('Customer')['Customer.Last.Day.Won'].fillna(method='ffill')
            self.data['Customer.Last.Day.Won'] = self.data.groupby('Customer')['Customer.Last.Day.Won'].shift(1)
            self.data['Customer.Days.Since.LastWin'] = (self.data['Create.Day'] - self.data['Customer.Last.Day.Won']).dt.days
            self.data.loc[idxWon, 'Customer.CHF.Last.Won'] = self.data.loc[idxWon, 'Order.Entry.CHF']
            self.data['Customer.CHF.Last.Won'] = self.data.groupby('Customer')['Customer.CHF.Last.Won'].fillna(method='ffill')
            self.data['Customer.CHF.Last.Won'] = self.data.groupby('Customer')['Customer.CHF.Last.Won'].shift(1)
            self.data.drop(['Customer.Last.Day.Won'], axis=1, inplace=True)

        def time_since_chf_last_loss():
            idxLost = (self.data.Target == 0)
            self.data.loc[idxLost, 'Customer.Last.Day.Loss'] = self.data.loc[idxLost, 'Create.Day']
            self.data['Customer.Last.Day.Loss'] = self.data.groupby('Customer')['Customer.Last.Day.Loss'].fillna(method='ffill')
            self.data['Customer.Last.Day.Loss'] = self.data.groupby('Customer')['Customer.Last.Day.Loss'].shift(1)
            self.data['Customer.Days.Since.LastLoss'] = (self.data['Create.Day'] - self.data['Customer.Last.Day.Loss']).dt.days
            self.data.loc[idxLost, 'Customer.CHF.Last.Loss'] = self.data.loc[idxLost, 'Order.Entry.CHF']
            self.data['Customer.CHF.Last.Loss'] = self.data.groupby('Customer')['Customer.CHF.Last.Loss'].fillna(method='ffill')
            self.data['Customer.CHF.Last.Loss'] = self.data.groupby('Customer')['Customer.CHF.Last.Loss'].shift(1)
            self.data.drop(['Customer.Last.Day.Loss'], axis=1, inplace=True)

        # pass
        first_contact()
        conv_rate_so_far()
        conv_rate_last_five()
        conv_rate_last_year()
        # stddev_offer_to_avg_contract_size()
        # last_outcome()
        # time_since_last_contact()
        # time_since_chf_last_win()
        # time_since_chf_last_loss()

    def _drop_one(self):
        '''
        Drops the dummy column of ones
        '''
        self.data.drop('one', axis=1, inplace=True)

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
        # self._attach_one()
        # self._create_time_units()
        self._create_target_variable()
        # self._create_customer_metrics()
        # self._drop_one()
        # self._drop_initial_bulk_load()
        return self.data
