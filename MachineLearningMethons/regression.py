import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
class regression:

    # Data preparation
    def __dataPre(self):

        dataset = pd.read_csv('./modelFiles/Social_Network_Ads.csv')
        X = dataset.iloc[:, 0:2 ].values
        y = dataset.iloc[:, -1].values
        return X, y

    def __featureScale(self):
        X, y = self.__dataPre()
        # Future Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X,y

    def __linearR(self):
        X, y = self.__featureScale()
        regressor = LinearRegression()
        regressor.fit(X, y)
        r_squared = regressor.score(X, y)
        print(f"LinearRegression: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = regressor.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Linear Regression : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __polyR(self):
        X, y = self.__featureScale()
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X, y)
        poly_reg = PolynomialFeatures(degree=7)
        X_poly = poly_reg.fit_transform(X)
        lin_reg_2 = LinearRegression()
        lin_reg_2.fit(X_poly, y)
        r_squared = regressor.score(X, y)
        print(f"PolynomialRegression: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = regressor.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Polynominal Regression : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __SVRr(self):
        X, y = self.__featureScale()
        regressor = SVR(kernel='rbf')
        regressor.fit(X, y)
        r_squared = regressor.score(X, y)
        print(f"SupportVectorMachine: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = regressor.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Support Vector Machine Regression : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __decisionTreeR(self):
        X, y = self.__featureScale()
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(X, y)
        r_squared = regressor.score(X, y)
        print(f"DecisionTreeRegressor: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = regressor.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Decision Tree Regressor : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __randomForestTreeR(self):
        X, y = self.__featureScale()
        regressor = RandomForestRegressor(n_estimators=10, random_state=1)
        regressor.fit(X, y)
        r_squared = regressor.score(X, y)
        print(f"RandomForestRegressor: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = regressor.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Random Forest Regressor : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __logisticR(self):
        X, y = self.__featureScale()
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X, y)
        r_squared = classifier.score(X,y)
        print((f"LogisticRegression: {r_squared}"))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Logistic Regression Classification: {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def callFunction(self):
        print("this is linear regression result \n")
        self.__linearR()
        print("this is multiple linear regression result \n")
        self.__polyR()
        print("this is support vektor machine regression result \n")
        self.__SVRr()
        print("this is decision tree regression result \n")
        self.__decisionTreeR()
        print("this is random forest regression result \n")
        self.__randomForestTreeR()









