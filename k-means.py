import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

MAX_CLUSTER_NUMBER = 11
COLORS = cm.rainbow(np.linspace(0, 1, MAX_CLUSTER_NUMBER))
LABELS = ['cluster '+ str(i) for i in range(MAX_CLUSTER_NUMBER)]

dataset=pd.read_csv('Mall_Customers.csv', error_bad_lines=False)
print(dataset.head())

#Get attributes to use
X = dataset.iloc[:, [2, 4]].values

#Plot starting dataset
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend()
plt.title('Plot of data points')
plt.show()


m = X.shape[0] #Training examples
n = X.shape[1] #Features

number_of_iterations = 100 #Number of epochs

WCSS_array=np.array([]) #Array for optimization in k clusters

for iterations in range(1, MAX_CLUSTER_NUMBER):
    K = iterations #Number of clusters
    #Centroids for k-means
    Centroids=np.array([]).reshape(n,0) 

    #Initializing centroids to random value in dataset
    for i in range(K):
        rand=rd.randint(0,m-1)
        Centroids=np.c_[Centroids,X[rand]]


    Output={}
    for i in range(number_of_iterations):
        #Initialize euclidian distance array with zeros
        EuclidianDistance=np.array([]).reshape(m,0)
        #Get euclidian distance for each point in respect for each centroid
        for k in range(K):
            temp_for_distance=np.sum((X-Centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,temp_for_distance]
        #For each point in dataset get the closest centroid (the one with the minimum distance)
        ClusterAssigned=np.argmin(EuclidianDistance,axis=1)+1


        IterationRes={}
        #Initialize dictionary of outputs per cluster with empty arrays with the shape of the points of the dataset
        for k in range(K):
            IterationRes[k+1]=np.array([]).reshape(2,0)

        #For each point in the array of closests cluster to point add to the array belonging to the cluster
        for i in range(m):
            IterationRes[ClusterAssigned[i]]=np.c_[IterationRes[ClusterAssigned[i]],X[i]]

        #Transpose because of numpy
        for k in range(K):
            IterationRes[k+1]=IterationRes[k+1].T
        #Get the new centroids based on the mean for each collection of points belonging to each cluster
        for k in range(K):
            Centroids[:,k]=np.mean(IterationRes[k+1],axis=0)
        Output=IterationRes

    #Plot output of clustering
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=COLORS[k],label=LABELS[k])
    plt.scatter(Centroids[0,:],Centroids[1,:],s=100,c='yellow',label='Centroids')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.show()

    #Optimization of k clusters, get the sum of squared distances between centroids and clustering result per iteration of k clusters
    wcss=0
    for k in range(K):
        wcss+=np.sum((Output[k+1]-Centroids[:,k])**2)
    WCSS_array=np.append(WCSS_array,wcss)



#Plot of Within-cluster sums of squares to get optimum number of clusters
ClusterArray=np.arange(1,MAX_CLUSTER_NUMBER,1)
plt.plot(ClusterArray, WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('Within-cluster sums of squares (WCSS)')
plt.title('Elbow method to determine optimum number of clusters')
plt.show()