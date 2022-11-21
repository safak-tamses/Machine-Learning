from MachineLearningMethons import clustering
from MachineLearningMethons import regression
from MachineLearningMethons import classification
def main():
    clusteringObject = clustering.cluster()
    clusteringObject.callFunction()

    regressionObject = regression.regression()
    regressionObject.callFunction()

    classificationObject = classification.classification()
    classificationObject.callFunction()






if __name__ == "__main__":
    main()