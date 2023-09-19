import sys
import os
# Hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import data_reader
df = data_reader.DataReader().pd_read('chip_dataset.csv')
print(df.head())