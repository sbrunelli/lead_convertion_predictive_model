import numpy as np
from data_cleaning import DataCleaner

dc = DataCleaner()
data = dc.clean()

columns = [
# 'Create.Day',
'Customer',
'Target',
'Customer.Contacts.So.Far',
# 'Customer.First.Contact',
'Customer.Won.So.Far',
'Customer.ConvRatio.So.Far']
data = data[columns]

print data.head(20)
