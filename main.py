from MachineLearningMethons import clustering
from MachineLearningMethons import regression
from MachineLearningMethons import classification
from MachineLearningMethons import artificialNeuralNetwork
def main():
    annObject = artificialNeuralNetwork.aNN()
    annObject.callFunction()

    clusteringObject = clustering.cluster()
    clusteringObject.callFunction()

    regressionObject = regression.regression()
    regressionObject.callFunction()

    classificationObject = classification.classification()
    classificationObject.callFunction()



if __name__ == "__main__":
    main()