import os,sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier_base.cbase import Classifier_Base
from sklearn import ensemble

class RandomForest(Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)
    
    def train(self, mode) :
        rf_clf = ensemble.RandomForestClassifier(n_estimators=100,
                                                 bootstrap=True,
                                                 oob_score=True,
                                                 criterion='gini',
                                                 max_depth=1,
                                                 min_samples_leaf=2,
                                                 ccp_alpha=0.0)
        rf_clf.fit(self.X_train, self.y_train)
        print("Test score : ", rf_clf.score(self.X_test, self.y_test))
        print("OOB score : ", rf_clf.oob_score_)
        return rf_clf

# RandomForest('chip_dataset_cleaned.csv', 'binary')
RandomForest('chip_dataset_cleaned.csv', 'multi')