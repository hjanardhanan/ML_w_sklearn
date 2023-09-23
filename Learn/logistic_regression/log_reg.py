# Do clustering to :
# 1. Predict if CPU or GPU (Binary classifier)
# 2. Predict vendor  (multi-class classifier)
# 3. Predict the Foundry (multi-class classifier)

import sys
import os
# Hack
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data import data_reader
df = data_reader.DataReader().pd_read('chip_dataset.csv')
print(df.head())