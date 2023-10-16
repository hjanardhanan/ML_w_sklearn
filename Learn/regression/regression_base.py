import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

class RegressionBase :
    def __init__(self, filename) :
        df = pd.read_csv('./Learn/regression/data/' + filename)
        self.test_train_split(df)
        self.target_col = ['price']
        self.train_cols = list(set(df.columns) - set(self.target_col))

    def test_train_split(self, df) :
        print ("Test & train split ..")
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(df.loc[:, self.train_cols], df.loc[:, self.target_col], \
                                random_state=203, test_size=0.2, shuffle=True)

    def evaluate(self, model) :
        print ("Evaluating regression model ..")

RegressionBase('scrap_price_cleaned.csv')