# Do clustering to :
# 1. Predict Type, if CPU or GPU (Binary classifier)
# 2. Predict the Foundry (multi-class classifier)

# Hack
import os, sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier_base import cbase
from sklearn import linear_model

class LogisticReg(cbase.Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)
    
    def train(self, mode) :
        print ("Training .. ")
        model = linear_model.LogisticRegression(random_state=22).fit(self.X_train, self.y_train)
        return model
   
# Test 
# LogisticReg('chip_dataset_cleaned.csv', 'binary')
LogisticReg('chip_dataset_cleaned.csv', 'multi')