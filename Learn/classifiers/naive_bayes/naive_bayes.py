import os,sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier_base.cbase import Classifier_Base
from sklearn.naive_bayes import GaussianNB, MultinomialNB

class NaiveBayes(Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)
    
    def train(self, mode) :
        gauss_clf = GaussianNB(
                        priors=None,
                        var_smoothing=1e-9
                    ).fit(self.X_train, self.y_train)
        
        multi_clf = MultinomialNB(
                        alpha=1,
                        fit_prior=True,
                        class_prior=None
                    ).fit(self.X_train, self.y_train)
        
    
        # Print some info

        return clf