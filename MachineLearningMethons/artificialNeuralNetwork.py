# Artificial Neural Network
import numpy
# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
class aNN():


    def __readData(self):
        dataset = pd.read_csv("./modelFiles/churnModelling.csv")
        self.__X = dataset.iloc[:, 3:-1].values
        self.__y = dataset.iloc[:, -1].values

    def __encodingData(self):
        self.__readData()
        le = LabelEncoder()
        self.__X[:, 2] = le.fit_transform(self.__X[:, 2])
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        self.__X = np.array(ct.fit_transform(self.__X))
    def __splitModel(self):
        self.__encodingData()
        from sklearn.model_selection import train_test_split
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.__X, self.__y, test_size = 0.2, random_state = 0)
    def __featureScaling(self):
        self.__splitModel()
        sc = StandardScaler()
        self.__X_train = sc.fit_transform(self.__X_train)
        self.__X_test = sc.transform(self.__X_test)
    def __createANN(self):
        self.__featureScaling()
        sc = StandardScaler()
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the second hidden layer
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        # Part 3 - Training the ANN
        # Compiling the ANN
        ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        # Training the ANN on the Training set
        ann.fit(self.__X_train, self.__y_train, batch_size = 32, epochs = 100)
        # Part 4 - Making the predictions and evaluating the model
        # Predicting the result of a single observation

        # Predicting the Test set results
        y_pred = ann.predict(self.__X_test)
        y_pred = (y_pred > 0.5)
        print(np.concatenate((y_pred.reshape(len(y_pred),1), self.__y_test.reshape(len(self.__y_test),1)),1))
        # Making the Confusion Matrix
        cm = confusion_matrix(self.__y_test, y_pred)
        print(cm)
        accuracy_score(self.__y_test, y_pred)
    def callFunction(self):
        return self.__createANN()


temp = aNN()
temp.callFunction()
