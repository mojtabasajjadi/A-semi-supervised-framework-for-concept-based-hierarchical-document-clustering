# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Optional word summarization.

############################## Import librerie's
import numpy as np
import pickle

import time

""""""""""""""""""""""""""""""""""""""
"              INPUT                 "
""""""""""""""""""""""""""""""""""""""
start_time = time.time()

open_file = open("x_l.pkl", "rb")
x_l = pickle.load(open_file)
open_file.close()

open_file = open("y_l.pkl", "rb")
y_l = pickle.load(open_file)
open_file.close()

open_file = open("v_l.pkl", "rb")
v_l = pickle.load(open_file)
open_file.close()

open_file = open("x_u.pkl", "rb")
x_u = pickle.load(open_file)
open_file.close()

open_file = open("y_u.pkl", "rb")
y_u = pickle.load(open_file)
open_file.close()

open_file = open("v_u.pkl", "rb")
v_u = pickle.load(open_file)
open_file.close()



""""""""""""""""""""""""""""""""""""""
"              MAIN                  "
""""""""""""""""""""""""""""""""""""""
def prePross(x,y):
    x2 = np.concatenate( x , axis=0 )
    y2 = []
    for i in range(len(y)):
        y2.append( y[i].rstrip())
    return x2 , y2

# convert list of data's to matrix numpy to use in k-means clustering
x_u , y_u= prePross(x_u , y_u)
x_l , y_l= prePross(x_l , y_l)
x = np.concatenate( ( x_l , x_u ) , axis=0 )
y = y_l + y_u
v = v_l + v_u


num_clusters = 2000
from sklearn.cluster import KMeans
kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42 )
kmeans.fit(x)

# The lowest SSE value
sse =  kmeans.inertia_
# Final locations of the centroid
centroid_kmeans =  kmeans.cluster_centers_
# Labels
label_kmeans = kmeans.labels_



""""""""""""""""""""""""""""""""""""""
"           BUILD CLUSTERS           "
""""""""""""""""""""""""""""""""""""""
#define clusters by considering lables and data
clusters = []

for i in range(num_clusters):
    temp = np.where(label_kmeans == i)
    c = np.reshape( x[temp[0][0]] , (1,400)  )
    for j in range(len(temp[0]) -1 ):     
        c = np.concatenate( ( c , np.reshape(x[temp[0][j+1]],(1,400)) ) , axis=0 )
    clusters.append(c)

labels = []    
vocabs = []
for i in range(num_clusters):
    temp = np.where(label_kmeans == i)
    vt = []
    l = []
    for j in range(len(temp[0])):
        vt.append(v[temp[0][j]])
        l.append(y[temp[0][j]])
    vocabs.append(vt)
    labels.append(l)



""""""""""""""""""""""""""""""""""""""
"       PROSECC UNPURE CLUSTER       "
"""""""""""""""""""""""""""""""""""""" 
# this section split unpure clusters into two pure clusters in aspect of labels
def addNewCluster(i,agr,crd,trd,nts,unl):
    if agr[0].size != 0 :
        temp_vocab = []
        temp_cluster = np.reshape( clusters[i][agr[0][0]] , (1,400))
        temp_label = labels[i][agr[0][0]]
        for k in range(agr[0].size):
            j = agr[0][k]
            if k!=0:
                temp_cluster = np.concatenate(( temp_cluster , np.reshape( clusters[i][j] , (1,400))) , axis=0)
            temp_vocab.append(vocabs[i][j])
        final_cluster.append(temp_cluster)
        final_labels.append(temp_label)
        final_vocabs.append(temp_vocab)

    if crd[0].size != 0 :
        temp_vocab = []
        temp_cluster = np.reshape( clusters[i][crd[0][0]] , (1,400))
        temp_label = labels[i][crd[0][0]]
        for k in range(crd[0].size):
            j = crd[0][k]
            if k!=0:
                temp_cluster = np.concatenate(( temp_cluster , np.reshape( clusters[i][j] , (1,400))) , axis=0)
            temp_vocab.append(vocabs[i][j])
        final_cluster.append(temp_cluster)
        final_labels.append(temp_label)
        final_vocabs.append(temp_vocab)

    if trd[0].size != 0 :
        temp_vocab = []
        temp_cluster = np.reshape( clusters[i][trd[0][0]] , (1,400))
        temp_label = labels[i][trd[0][0]]
        for k in range(trd[0].size):
            j = trd[0][k]
            if k!=0:
                temp_cluster = np.concatenate(( temp_cluster , np.reshape( clusters[i][j] , (1,400))) , axis=0)
            temp_vocab.append(vocabs[i][j])
        final_cluster.append(temp_cluster)
        final_labels.append(temp_label)
        final_vocabs.append(temp_vocab)

    if nts[0].size != 0 :
        temp_vocab = []
        temp_cluster = np.reshape( clusters[i][nts[0][0]] , (1,400))
        temp_label = labels[i][nts[0][0]]
        for k in range(nts[0].size):
            j = nts[0][k]
            if k!=0:
                temp_cluster = np.concatenate(( temp_cluster , np.reshape( clusters[i][j] , (1,400))) , axis=0)
            temp_vocab.append(vocabs[i][j])
        final_cluster.append(temp_cluster)
        final_labels.append(temp_label)
        final_vocabs.append(temp_vocab)

    if unl[0].size != 0 :
        temp_vocab = []
        temp_cluster = np.reshape( clusters[i][unl[0][0]] , (1,400))
        temp_label = labels[i][unl[0][0]]
        for k in range(unl[0].size):
            j = unl[0][k]
            if k!=0:
                temp_cluster = np.concatenate(( temp_cluster , np.reshape( clusters[i][j] , (1,400))) , axis=0)
            temp_vocab.append(vocabs[i][j])
        final_cluster.append(temp_cluster)
        final_labels.append(temp_label)
        final_vocabs.append(temp_vocab)


def processClusters(i):
    lst = labels[i]
    lst = np.array(lst)
    agr = np.where(lst == "agriculture")
    crd = np.where(lst == "crude")
    trd = np.where(lst == "trade")
    nts = np.where(lst == "interest") 
    unl = np.where(lst == "unlabeled") 
    
    content = list(set(lst))
    if "unlabeled" in content : content.remove("unlabeled")
    
    if len(content) == 1:
        final_cluster.append(clusters[i])
        final_labels.append(labels[i][0])
        final_vocabs.append(vocabs[i])
    elif len(content) >= 2:
        addNewCluster(i,agr,crd,trd,nts,unl)


final_cluster = []
final_labels = []
final_vocabs = []



for i in range(len(labels)):
    length = len(list(set(labels[i])))
    if length == 1 :
       final_cluster.append(clusters[i])
       final_labels.append(labels[i][0])
       final_vocabs.append(vocabs[i])
    else:
       processClusters(i)



""""""""""""""""""""""""""""""""""""""
"              SAVE                  "
""""""""""""""""""""""""""""""""""""""
import pickle
filename_fc = "kmeans_fc.pkl" #k-means final clusters

filename_fl = "kmeans_fl.pkl" #k-means final lables

filename_fv = "kmeans_fv.pkl" #k-means final vocabularies


open_file = open(filename_fc, "wb")
pickle.dump(final_cluster, open_file)
open_file.close()


open_file = open(filename_fl, "wb")
pickle.dump(final_labels, open_file)
open_file.close()


open_file = open(filename_fv, "wb")
pickle.dump(final_vocabs, open_file)
open_file.close()


print(" %s seconds" % (time.time() - start_time))


