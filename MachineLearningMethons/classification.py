import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class classification:

    # Data preparation
    def __dataPre(self):
        dataset = pd.read_csv('./modelFiles/Social_Network_Ads.csv')
        X = dataset.iloc[:, 0:2 ].values
        y = dataset.iloc[:, -1].values
        return X, y

    def __featureScale(self):
        X, y = self.__dataPre()
        # Future Scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X,y

    def __kNeighborsC(self):
        X, y = self.__featureScale()
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"kNearstNeighbors: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"KNeighbors Classification : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __SVClinear(self):
        X, y = self.__featureScale()
        classifier = SVC(kernel='linear', random_state=0)
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"SupportVectorMachine: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Support Vector Classification (kernel=Linear) : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __SVCrbf(self):
        X, y = self.__featureScale()
        classifier = SVC(kernel='rbf', random_state=0)
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"KernelSupportVectorMachine: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Support Vector Classification (kernel=RBF) : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __naiveBayesC(self):
        X, y = self.__featureScale()
        classifier = GaussianNB()
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"NaiveBayes: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Naive Bayes Classification : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __decisionTreeC(self):
        X, y = self.__featureScale()
        classifier = DecisionTreeClassifier(criterion='entropy', random_state=1)
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"DecisionTreeClassification: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Decision Tree  Classification : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def __randomForestC(self):
        X, y = self.__featureScale()
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=1)
        classifier.fit(X, y)
        r_squared = classifier.score(X, y)
        print(f"RandomForestClassification: {r_squared}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
        y_predicted = classifier.predict(X_test)
        y_predicted = np.round(y_predicted)
        conMatrix = confusion_matrix(y_test, y_predicted)
        print(f"Random Forest Classifier : {conMatrix}")
        trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
        falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
        confusionMatrixScore = (trueValue / (trueValue + falseValue))
        print(confusionMatrixScore[0][0])
        print("\n")

    def callFunction(self):
        print("this is k neighbors classification result \n")
        self.__kNeighborsC()
        print("this is support vector classification (kernel= linear) result \n")
        self.__SVClinear()
        print("this is support vector classification (kernel= rbf) result \n")
        self.__SVCrbf()
        print("this is naive bayes classification result \n")
        self.__naiveBayesC()
        print("this is decision tree classification result \n")
        self.__decisionTreeC()
        print("this is random forest classification result \n")
        self.__randomForestC()

