import numpy as np
import pandas as pd



# Data preparation
def dataPre():

    dataset = pd.read_csv('../../SKLearn-Machine-Learning/Social_Network_Ads.csv')
    X = dataset.iloc[:, 0:2 ].values
    y = dataset.iloc[:, -1].values
    return X, y

def featureScale():
    X, y = dataPre()
    # Future Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X,y

def linearR():
    X, y = featureScale()
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X, y)
    r_squared = regressor.score(X, y)
    print(f"LinearRegression: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = regressor.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Linear Regression : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def polyR():
    X, y = featureScale()
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = regressor.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Polynominal Regression : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def SVRr():
    X, y = featureScale()
    from sklearn.svm import SVR
    regressor = SVR(kernel='rbf')
    regressor.fit(X, y)
    r_squared = regressor.score(X, y)
    print(f"SupportVectorMachine: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = regressor.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Support Vector Machine Regression : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def decisionTreeR():
    X, y = featureScale()
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=1)
    regressor.fit(X, y)
    r_squared = regressor.score(X, y)
    print(f"DecisionTreeRegressor: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = regressor.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Decision Tree Regressor : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def randomForestTreeR():
    X, y = featureScale()
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=10, random_state=1)
    regressor.fit(X, y)
    r_squared = regressor.score(X, y)
    print(f"RandomForestRegressor: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = regressor.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Random Forest Regressor : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def logisticR():
    X, y = featureScale()
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X, y)
    r_squared = classifier.score(X,y)
    print((f"LogisticRegression: {r_squared}"))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Logistic Regression Classification: {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def kNeighborsC():
    X, y = featureScale()
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
    classifier.fit(X,y)
    r_squared =classifier.score(X, y)
    print(f"kNearstNeighbors: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"KNeighbors Classification : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def SVClinear():
    X, y = featureScale()
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear',random_state=0)
    classifier.fit(X, y)
    r_squared = classifier.score(X,y)
    print(f"SupportVectorMachine: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Support Vector Classification (kernel=Linear) : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def SVCrbf():
    X, y = featureScale()
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X, y)
    r_squared = classifier.score(X,y)
    print(f"KernelSupportVectorMachine: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Support Vector Classification (kernel=RBF) : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def naiveBayesC():
    X, y = featureScale()
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
    r_squared = classifier.score(X,y)
    print(f"NaiveBayes: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Naive Bayes Classification : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def decisionTreeC():
    X, y = featureScale()
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy',random_state=1)
    classifier.fit(X,y)
    r_squared = classifier.score(X,y)
    print(f"DecisionTreeClassification: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Decision Tree  Classification : {conMatrix}")
    trueValue = conMatrix[:1, :1] + conMatrix[-1, -1]
    falseValue = conMatrix[1:, :1] + conMatrix[:1, 1:]
    confusionMatrixScore = (trueValue / (trueValue + falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")

def randomForestC():
    X, y = featureScale()
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=1)
    classifier.fit(X, y)
    r_squared = classifier.score(X,y)
    print(f"RandomForestClassification: {r_squared}")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
    y_predicted = classifier.predict(X_test)
    y_predicted = np.round(y_predicted)
    from sklearn.metrics import confusion_matrix, accuracy_score
    conMatrix = confusion_matrix(y_test, y_predicted)
    print(f"Random Forest Classifier : {conMatrix}")

    trueValue = conMatrix[:1,:1] + conMatrix[-1,-1]
    falseValue = conMatrix[1:,:1] + conMatrix[:1,1:]
    confusionMatrixScore = (trueValue/(trueValue+falseValue))
    print(confusionMatrixScore[0][0])
    print("\n")








# linearR()
# polyR()
# SVRr()
# decisionTreeR()
# randomForestTreeR()
# logisticR()
# kNeighborsC()
# SVClinear()
# SVCrbf()
# naiveBayesC()
# decisionTreeC()
# randomForestC()