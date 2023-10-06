import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

# Hack
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data import data_reader

class Classifier_Base :
    def __init__(self, filename, mode) -> None:
         df = data_reader.DataReader().pd_read(filename)
         self.clean_n_split(df, mode)

    def clean_n_split(self, df, mode) :
        print ("Cleaning and splitting .. ")
        # Drop multi/binary label based on mode
        drop_col = 'Foundry' if mode == 'binary' else 'Type'
        target_cols = 'Foundry' if mode != 'binary' else 'Type'
        df.drop(drop_col, axis = 1, inplace = True)
        # Training data columns
        # TBD: Explore Stratified splitting
        train_cols = list(set(df.columns) - set([target_cols]))
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(df.loc[:, train_cols], df.loc[:, target_cols], \
                                random_state=203, test_size=0.2, shuffle=True)
    
    def evaluate(self, mode, model) :
        print ("Evaluate .. ")
        y_pred = model.predict(self.X_test)
        report = metrics.classification_report(self.y_test, y_pred)
        print (" \n------------------- Classification Report ---------------- \n")
        print(report)
        print (" ---------------------------------------------------------- \n")
        print("Classes : ", model.classes_)
        print (" ---------------------------------------------------------- \n")
        scores = cross_val_score(model, self.X_train, self.y_train, cv = 5)
        print("CV score = ", scores)


