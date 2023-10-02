# Knn and RadiusNeigborsClassifier
# Hack
import os,sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn import neighbors, metrics, model_selection
from classifier_base.cbase import Classifier_Base
import numpy as np

class Knn (Classifier_Base) :
    def __init__(self, filename, mode) -> None:
        super().__init__(filename, mode)
        model = self.train(mode)
        super().evaluate(mode, model)
    
    def train(self, mode) :
        scores = []
        for k in range(1, 15) :
            knn = neighbors.KNeighborsClassifier(n_neighbors=k)
            score = model_selection.cross_val_score(knn, self.X_train, self.y_train, cv = 5)
            scores.append(np.mean(score))
        best_k = 1 + np.argmax(scores)
        print ("Training with k = ", best_k)
        return neighbors.KNeighborsClassifier(n_neighbors = best_k).fit(self.X_train, self.y_train)

Knn('chip_dataset_cleaned.csv', 'multi')