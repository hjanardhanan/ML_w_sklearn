# Do clustering to :
# 1. Predict Type, if CPU or GPU (Binary classifier)
# 2. Predict the Foundry (multi-class classifier)

import sys
import os

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Hack
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import data_reader

class LogisticReg :
    def __init__(self, filename, mode) -> None:
        df = data_reader.DataReader().pd_read(filename)
        self.clean_n_split(df, mode)
        self.train(mode)
        self.evaluate(mode, self.predict(mode))
    
    def clean_n_split(self, df, mode) :
        print ("Cleaning and splitting .. ")
        # Drop multi/binary label based on mode
        drop_col = 'Foundry' if mode == 'binary' else 'Type'
        target_cols = 'Foundry' if mode != 'binary' else 'Type'
        df.drop(drop_col, axis = 1, inplace = True)
        # Training data columns
        train_cols = list(set(df.columns) - set([target_cols]))
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(df.loc[:, train_cols], df.loc[:, target_cols], \
                                random_state=203, test_size=0.2, shuffle=True)

    def train(self, mode) :
        print ("Training .. ")
        self.model = linear_model.LogisticRegression(random_state=22).fit(self.X_train, self.y_train)
    
    def predict(self, mode) :
        print ("Predicting .. ")
        y_pred = self.model.predict(self.X_test)
        return y_pred
    
    def evaluate(self, mode, y_pred) :
        print ("Evaluating .. ")
        acc = metrics.accuracy_score(self.y_test, y_pred)
        print ("Accuracy = ", acc*100)
    
# Test 
LogisticReg('chip_dataset_cleaned.csv', 'binary')
# LogisticReg('chip_dataset_cleaned.csv', 'multi')