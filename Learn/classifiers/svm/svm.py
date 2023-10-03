# Study different kernels : linear, polynomial, rbf
# Choose optimal C and gamma using grid search
# Hack
import os,sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn import svm
from classifier_base.cbase import Classifier_Base
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import precision_score, make_scorer

class Svc(Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)

    def train(self, mode) :
        print ("Training .. ")
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        degrees = list(range(1, 4))
        Cs = list(np.arange(0.5, 10.5, 1.0))
        parameters =    {
                            'kernel' : kernels,
                            'degree' : degrees,
                            'C' : Cs
                        }
        # precision needs more arguments which cannot be directly passed in 
        precision_wrapper = make_scorer(precision_score, average='micro')#, zero_division=np.nan)
        svc = GridSearchCV(estimator=svm.SVC(),
                            param_grid=parameters,
                                scoring='precision' if mode == 'binary' else precision_wrapper )
        svc.fit(self.X_train, self.y_train)
        print(svc.best_params_)
        print(svc.best_score_)
        print(svc.best_estimator_)
        return svc.best_estimator_ 


Svc('chip_dataset_cleaned.csv', 'multi')