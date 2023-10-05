import os,sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn import tree
from classifier_base.cbase import Classifier_Base
import matplotlib.pyplot as plt

class DTree(Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)
    
    def train(self, mode) :
        print ("Training .. ")
        dtc = tree.DecisionTreeClassifier(criterion='entropy',
                                          splitter='best',
                                          max_depth=None,
                                          min_samples_split=3, # smallest label has 2 observations
                                          min_samples_leaf=1, # 
                                          min_weight_fraction_leaf=0,
                                          max_features=7,
                                          max_leaf_nodes=None,
                                          random_state=33,
                                          min_impurity_decrease=0.0,
                                          class_weight=None,
                                          ccp_alpha=0.0)
        # Find ideal post pruning value for alpha
        path = dtc.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        global_score = None
        for ccp_alpha in ccp_alphas :
            dtc_local = tree.DecisionTreeClassifier(criterion='entropy',
                                          splitter='best',
                                          max_depth=None,
                                          min_samples_split=3, # smallest label has 2 observations
                                          min_samples_leaf=1, # 
                                          min_weight_fraction_leaf=0,
                                          max_features=7,
                                          max_leaf_nodes=None,
                                          random_state=33,
                                          min_impurity_decrease=0.0,
                                          class_weight=None,
                                          ccp_alpha=ccp_alpha).fit(self.X_train, self.y_train)
            sc = dtc_local.score(self.X_train, self.y_train)
            if global_score is None or \
                    sc > global_score :
                dtc = dtc_local
                global_score = sc
        return dtc
        
# DTree('chip_dataset_cleaned.csv', 'binary')
DTree('chip_dataset_cleaned.csv', 'multi')