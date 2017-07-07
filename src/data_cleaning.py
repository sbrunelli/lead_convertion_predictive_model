import pandas as pd
import re

class DataCleaner(object):

    def __init__(self):
        self.data = None

    def _read_data(self):
        self.data = pd.read_csv('./data/opportunities.csv')

    def _clean_order_entry_chf(self):
        self.data['Order.Entry.CHF'] = self.data['Order.Entry.CHF'].map(lambda x: float(re.sub(',','',x)))

    def _create_target_variable(self):
        self.data['target'] = self.data['Status.Category'].map(lambda x: 1 if x=='Won' else 0)

    def clean(self):
        self._read_data()
        self._clean_order_entry_chf()
        self._create_target_variable()
        return self.data
