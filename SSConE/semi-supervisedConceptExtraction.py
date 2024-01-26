# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Document Representation-> Semi-supervised concept extraction,


############################## Import librerie's
from scipy import sparse
from scipy.spatial import distance
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances


""""""""""""""""""""""""""""""""""""""
"              METHODS               "
""""""""""""""""""""""""""""""""""""""
def findClosestClusters(centroids):
    centroids_matrix =np.concatenate(centroids, axis=0)
    distance_matrix = pairwise_distances(centroids_matrix, metric='euclidean')
    distance_matrix = np.where(distance_matrix==0, 999999, distance_matrix)
#    print(np.unravel_index(distance_matrix.argmin(), distance_matrix.shape))
    row_idx, col_idx = distance_matrix.argmin()//distance_matrix.shape[1], distance_matrix.argmin()%distance_matrix.shape[1]
#    print(distance_matrix[ri][ci])
    return (row_idx , col_idx)

def newCluster(i,j,label):
    clusters.append(np.concatenate((clusters[i], clusters[j]), axis=0))
    temp = np.concatenate((clusters[i], clusters[j]), axis=0)
    temp_mean = temp.mean(axis=0)
    temp_centroid = np.reshape(temp_mean, (1, vector_size))
    centroids.append(temp_centroid)
    labels.append(label)
    status.append("unidentified")
    docs.append(docs[i] + docs[j] )

def mergeClusters(Ci , Cj):
    if labels[Ci] == "unlabeled" and labels[Cj] == "unlabeled":
        newCluster(Ci,Cj,"unlabeled") 
    elif labels[Ci] == "unlabeled" or labels[Cj] == "unlabeled" :
        if labels[Ci] == "unlabeled":
            newCluster(Ci,Cj,labels[Cj]) 
        if labels[Cj] == "unlabeled":
            newCluster(Ci,Cj,labels[Ci]) 
    elif labels[Ci] != "unlabeled" and labels[Cj] != "unlabeled" and labels[Ci] == labels[Cj]:
        newCluster(Ci,Cj,labels[Ci]) 

def removeClusters(Ci , Cj):
    if Ci < Cj:
        del clusters[Ci]
        del clusters[Cj-1]
        del labels[Ci]
        del labels[Cj-1]
        del centroids[Ci]
        del centroids[Cj-1]
        del status[Ci]
        del status[Cj-1]
        del docs[Ci]
        del docs[Cj-1]        
    if Cj < Ci:
        del clusters[Cj]
        del clusters[Ci-1]
        del labels[Cj]
        del labels[Ci-1]
        del centroids[Cj]
        del centroids[Ci-1]
        del status[Cj]
        del status[Ci-1]
        del docs[Cj]
        del docs[Ci-1]
     
def add2UniqClusters(Ci,Cj):
    uniq_clusters.append(clusters[Ci])
    uniq_centroids.append(centroids[Ci])
    uniq_labels.append(labels[Ci])
    uniq_status.append("identified")
    uniq_docs.append(docs[Ci])
    
    uniq_clusters.append(clusters[Cj])
    uniq_centroids.append(centroids[Cj])
    uniq_labels.append(labels[Cj])
    uniq_status.append("identified")
    uniq_docs.append(docs[Cj])
    removeClusters(Ci,Cj)


""""""""""""""""""""""""""""""""""""""
"              INPUT                 "
"      start main procedure          "
""""""""""""""""""""""""""""""""""""""
import pickle


open_file = open("kmeans_fc.pkl", "rb")
fc = pickle.load(open_file)
open_file.close()

open_file = open("kmeans_fl.pkl", "rb")
fl = pickle.load(open_file)
open_file.close()

open_file = open("kmeans_fv.pkl", "rb")
fv = pickle.load(open_file)
open_file.close()

X_train = fc
y_train = fl
v_train = fv






""""""""""""""""""""""""""""""""""""""
"             INITIALIZE             "
""""""""""""""""""""""""""""""""""""""
# Construct uniq clusters
uniq_clusters = []
uniq_centroids = []
uniq_labels = []
uniq_status = []
uniq_docs = []

vector_size = 400


# Construct clusters
clusters = []
centroids = []
labels = []
status = []
docs = []
for i in range(len(X_train)):
    clusters.append(X_train[i])
    labels.append(y_train[i])
    # Set the mean of each cluster to it's centroid
    temp = X_train[i]
    temp_mean = temp.mean(axis=0)
    temp_centroid = np.reshape(temp_mean, (1, vector_size))
    
    centroids.append(temp_centroid)
    status.append("unidentified")
    docs.append(v_train[i])

# Add Ci to Cluster candidte set T;
T = clusters
   




""""""""""""""""""""""""""""""""""""""
"             CLUSTERING             "
""""""""""""""""""""""""""""""""""""""

finish_flag = False
iteration_num = 0
while finish_flag == False :
    count = 0
    start_time = time.time()
    print("------------------")
    print("*Iteration:" + str(iteration_num))
    for i in range(len(clusters)):
        if status[i] == "unidentified" and labels[i] != "unlabeled":
            count = count +1
    if count > 1 :
        cluster_i , cluster_j = findClosestClusters(centroids)
        if labels[cluster_i] != "unlabeled" and labels[cluster_j] != "unlabeled" and labels[cluster_i] != labels[cluster_j]:
            add2UniqClusters(cluster_i,cluster_j)
        else:
            mergeClusters(cluster_i , cluster_j)
            removeClusters(cluster_i,cluster_j)
    else:
        print("outside")
        finish_flag = True
        break
    iteration_num = iteration_num +1
    print(" %s seconds" % (time.time() - start_time))



""""""""""""""""""""""""""""""""""""""
"             OUTPUT                 "
""""""""""""""""""""""""""""""""""""""
out_cluster = []
out_centroid = []
out_docs = []
out_labels = []
out_status = []
for item in range(len(uniq_clusters)):
    if uniq_clusters[item].shape[0] > 3:
        out_cluster.append(uniq_clusters[item])
        out_centroid.append(uniq_centroids[item])
        out_docs.append(uniq_docs[item])
        out_labels.append(uniq_labels[item])
        out_status.append(uniq_status[item])


""""""""""""""""""""""""""""""""""""""
"             SAVE                   "
""""""""""""""""""""""""""""""""""""""
import pickle
filename_centroid = "out_centroid.pkl"
filename_cluster = "out_cluster.pkl"
filename_docs = "out_docs.pkl"
filename_labels = "out_labels.pkl"


open_file = open(filename_centroid, "wb")
pickle.dump(out_centroid, open_file)
open_file.close()

open_file = open(filename_cluster, "wb")
pickle.dump(out_cluster, open_file)
open_file.close()

open_file = open(filename_docs, "wb")
pickle.dump(out_docs, open_file)
open_file.close()

open_file = open(filename_labels, "wb")
pickle.dump(out_labels, open_file)
open_file.close()















