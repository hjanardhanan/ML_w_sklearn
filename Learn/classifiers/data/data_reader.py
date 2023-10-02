import sys, os
import pandas as pd
import numpy as np

class DataReader :
    def pd_read(self, filename) :
        one_up_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file = os.path.join(one_up_path, 'data', filename)
        return pd.read_csv(file)