from kneed import KneeLocator
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
class cluster:
    def __dataPre(self):
        dataset = pd.read_csv('./modelFiles/50_Startups.csv')
        X = dataset.iloc[:, 0:3].values
        y = dataset.iloc[:, -1].values
        return X, y

    def __featureScale(self):
        X, y = self.__dataPre()
        # Future Scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        return X, y

    # to find the number of clusters
    def __elbowMethod(self):
        X, y = self.__featureScale()
        wcss = []
        for i in range(1, 30):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=1)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        x = range(0, len(wcss))
        kn = KneeLocator(x, wcss, curve='convex', direction='decreasing')
        plt.xlabel('number of clusters k')
        plt.ylabel('Sum of squared distances')
        plt.plot(x, wcss, 'bx-')
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        plt.show()
        return kn.knee

    # to find the number of clusters
    def __dendrogram(self):
        X, y = self.__featureScale()
        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()

    def __kMeansClustering(self):
        X, y = self.__featureScale()
        numberOfClusters = self.__elbowMethod()
        kmeans = KMeans(n_clusters=numberOfClusters, init='k-means++', random_state=2)
        kmeans.fit(X)
        y_kmeans = kmeans.fit_predict(X)
        # Visualising the clusters
        color = ['r', 'b', 'g', 'c', 'm', 'k']
        for k in range(numberOfClusters):
            arrayDot = X[y_kmeans == k, 0], X[y_kmeans == k, 1]
            plt.scatter(X[y_kmeans == k, 0], X[y_kmeans == k, 1], s=10, c=color[k % 6])
            plt.scatter((kmeans.cluster_centers_[k][0]), (kmeans.cluster_centers_[k][1]), s=20, c=color[k % 6])
            for i in range(len(arrayDot[0])):
                x1 = []
                y1 = []
                x2 = []
                y2 = []
                x2.append(arrayDot[0][i])
                y2.append(arrayDot[1][i])
                x1.append(kmeans.cluster_centers_[k][0])
                y1.append(kmeans.cluster_centers_[k][1])
                x3 = x1 + x2
                y3 = y1 + y2
                plt.plot(x3, y3, color[k % 6])
        plt.title('  ')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

    def __hierarchicalClustering(self):
        X, y = self.__featureScale()
        numberOfClusters = self.__elbowMethod()
        hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        y_hc = hc.fit_predict(X)
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()

    def callFunction(self):
        self.__hierarchicalClustering()
        self.__kMeansClustering()
