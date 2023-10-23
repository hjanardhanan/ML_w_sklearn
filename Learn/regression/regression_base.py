import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import os

class RegressionBase :
    def __init__(self, filename) :
        curr_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        file = os.path.join(curr_path, 'data', filename)
        df = pd.read_csv(file)
        self.target_col = ['passthru_price__price']
        self.train_cols = list(set(df.columns) - set(self.target_col))
        self.test_train_split(df)

    def test_train_split(self, df) :
        print ("Test & train split ..")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
                          (df.loc[:, self.train_cols], \
                            df.loc[:, self.target_col], \
                              random_state=203, test_size=0.2, shuffle=True)

    def evaluate(self, model) :
        print ("Evaluating regression model ..")
        y_pred = model.predict(self.X_test)
        scores = dict()
        scores['mse'] = metrics.mean_squared_error(y_pred, self.y_test)
        scores['rmse'] = metrics.mean_squared_error(y_pred, self.y_test, squared=False)
        scores['mae'] = metrics.mean_absolute_error(y_pred, self.y_test)
        print (" ========= SCORES ======== ")
        print (scores)
