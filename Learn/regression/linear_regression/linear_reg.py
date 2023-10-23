import os, sys
print (os.getcwd())
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path :
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regression_base import RegressionBase
from sklearn import linear_model

class LinearRegression(RegressionBase) :
    def __init__(self, filename):
        super().__init__(filename)
        model = self.build_model()
        super().evaluate(model)

    def build_model(self) :
        print ("Building model ..")
        model = linear_model.LinearRegression().fit(self.X_train, self.y_train)
        print ("-> Model score = ", model.score(self.X_test, self.y_test))
        return model

LinearRegression('scrap_price_cleaned.csv')