# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Split data into test and train, labled and unlabled train data


############################## Import librerie's

from sklearn.model_selection import train_test_split
from scipy import sparse
import numpy as np
#read document from csr matrix
document_vectors = sparse.load_npz("save/boc_matrix.npz")
num_documents = document_vectors.shape[0]
#read labels of document from text file
document_labels = []
with open('dataset/Reuters4class_label.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line
        # add item to the list
        document_labels.append(current_place)  
filehandle.close






X = document_vectors.toarray()
y = document_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

boc_dim = 400
num_labeled = 200


import random
idx = random.sample(range(len(y_train)), num_labeled)

x_train_u = []
y_train_u = []
x_train_l = []
y_train_l = []
for item in range(len(y_train)): 
    if item in idx: 
        x_train_l.append(np.reshape(X_train[item], (1, boc_dim)))
        y_train_l.append(y_train[item]) 
    else:
        x_train_u.append(np.reshape(X_train[item], (1, boc_dim)))
        y_train_u.append("unlabeled")
        


import pickle

filename_x_train_u = "x_train_u.pkl"
filename_x_train_l = "x_train_l.pkl"
filename_y_train_u = "y_train_u.pkl"
filename_y_train_l = "y_train_l.pkl"
filename_x_test = "x_test.pkl"
filename_y_test = "y_test.pkl"

open_file = open(filename_x_train_u, "wb")
pickle.dump(x_train_u, open_file)
open_file.close()

open_file = open(filename_x_train_l, "wb")
pickle.dump(x_train_l, open_file)
open_file.close()

open_file = open(filename_y_train_u, "wb")
pickle.dump(y_train_u, open_file)
open_file.close()

open_file = open(filename_y_train_l, "wb")
pickle.dump(y_train_l, open_file)
open_file.close()

open_file = open(filename_x_test, "wb")
pickle.dump(X_test, open_file)
open_file.close()

open_file = open(filename_y_test, "wb")
pickle.dump(y_test, open_file)
open_file.close()

