class FeatureExtractor(object):

    def __init__(self):
        self.data = None

    def _subset_columns(self):
        columns = ['Order.Entry.CHF']
        self.data = self.data[columns]

    def featurize(self, data):
        self.data = data
        self._subset_columns()
        return self.data.values
