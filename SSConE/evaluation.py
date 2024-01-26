# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Creating document vectors,
#   Classification and Clustering evaluation.



############################## Import librerie's
import numpy as np
from scipy.sparse import csr_matrix
""""""""""""""""""""""""""""""""""""""
"              INPUT                 "
""""""""""""""""""""""""""""""""""""""
import pickle


open_file = open("out_centroid.pkl", "rb")
out_centroid = pickle.load(open_file)
open_file.close()

open_file = open("out_cluster.pkl", "rb")
out_cluster = pickle.load(open_file)
open_file.close()

open_file = open("out_docs.pkl", "rb")
out_docs = pickle.load(open_file)
open_file.close()

open_file = open("out_labels.pkl", "rb")
out_labels = pickle.load(open_file)
open_file.close()


""""""""""""""""""""""""""""""""""""""
"      start main procedure          "
""""""""""""""""""""""""""""""""""""""
words_w2v = []
words_label = []
idx2word = []
c = 0
for i in range(len(out_cluster)):
    for j in range(len(out_cluster[i])):
        words_w2v.append(np.reshape(out_cluster[i][j],(1,400)))
        words_label.append(i)
        idx2word.append(c)
        c = c+1

words = []
for u in range(len(out_docs)):       
    ws = out_docs[u]
    xx = [x for x in ws if x]
    for k in range(len(xx)):
        words.append(xx[k])



""""""""""""""""""""""""""""""""""""""
"             Methods                "
""""""""""""""""""""""""""""""""""""""
def _create_w2c(idx2word, cluster_label, num_concept):
    if len(idx2word)!=len(cluster_label):
        raise IndexError("Dimensions between words and labels mismatched")
    rows=[i for i, idx2word in enumerate(idx2word)]
    cols=[j for j in cluster_label]
    vals=[1.0 for i in idx2word]
    return csr_matrix((vals, (rows, cols)), shape=(len(idx2word), num_concept))


from collections import Counter
def _create_bow(idx2word, doc_path):
    rows=[]
    cols=[]
    vals=[]

    word2idx = {word:idx for word, idx in zip(words, idx2word)}
#    print(word2idx)
    with open(doc_path, "r", encoding='utf-8') as f:
        for i, doc in enumerate(f):
            tokens=doc.rstrip().split(" ")
            tokens_count=Counter([word2idx[token] for token in tokens if token in word2idx])
            for idx, count in tokens_count.items():
                rows.append(i)
                cols.append(idx)
                vals.append(float(count))
    return csr_matrix((vals, (rows, cols)), shape=(i+1, len(word2idx)))

import scipy.sparse
from sklearn.utils.extmath import safe_sparse_dot
def _apply_cfidf(csr_matrix):
    num_docs, num_concepts=csr_matrix.shape
    _, nz_concept_idx=csr_matrix.nonzero()
    cf=np.bincount(nz_concept_idx, minlength=num_concepts)
    icf=np.log(num_docs / cf)
    icf[np.isinf(icf)]=0
    return safe_sparse_dot(csr_matrix, scipy.sparse.diags(icf))



""""""""""""""""""""""""""""""""""""""
"             MAIN                   "
""""""""""""""""""""""""""""""""""""""
# Construct concept vectors for documnets
open_file = open("x_test.pkl", "rb")
x_test = pickle.load(open_file)
open_file.close()

open_file = open("y_test.pkl", "rb")
y_test = pickle.load(open_file)
open_file.close()

y_test2 = []
for i in range(len(y_test)):
    y_test2.append( y_test[i].rstrip())
y_test = y_test2


with open('x_test.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(x_test))

matris_concept = _create_w2c(idx2word , words_label , len(out_cluster))
doc_path = 'x_test.txt'
matris_bow = _create_bow(idx2word , doc_path)
matris_boc=_apply_cfidf(safe_sparse_dot(matris_bow, matris_concept))

boc = csr_matrix(matris_boc, shape=(len(x_test), 400)).toarray()
max_index_col = np.argmax(boc, axis=1)
y_pred = []
for item in max_index_col:
    y_pred.append(out_labels[item])

y_pred_extra = []
for i in range(boc.shape[0]):
    temp_c = []
    for j in range(boc.shape[1]):
        if boc[i][j] != 0 :
            temp_c.append(out_labels[j])
    y_pred_extra.append(temp_c)




''''''''''''''''''''''''''''''''''''''''''''''''
'''            classification METHOD         '''
''''''''''''''''''''''''''''''''''''''''''''''''
## REUTERS 21578
y_pred_new = []
for i in range(boc.shape[0]):
    temp_c = []
    agr_w  = 0
    crd_w = 0
    trd_w = 0
    nts_w = 0    
    for j in range(boc.shape[1]):
        if boc[i][j] != 0 :
            if out_labels[j] == 'agriculture':
                agr_w = agr_w + boc[i][j]
            elif out_labels[j] == 'crude':
                crd_w = crd_w + boc[i][j]
            elif out_labels[j] == 'trade':
                trd_w = trd_w + boc[i][j]
            elif out_labels[j] == 'interest':
                nts_w = nts_w + boc[i][j]
    if max(agr_w,crd_w,trd_w,nts_w) == agr_w: y_pred_new.append('agriculture')
    elif max(agr_w,crd_w,trd_w,nts_w) == crd_w: y_pred_new.append('crude')
    elif max(agr_w,crd_w,trd_w,nts_w) == trd_w: y_pred_new.append('trade')
    elif max(agr_w,crd_w,trd_w,nts_w) == nts_w: y_pred_new.append('interest')


## ACC
from sklearn.metrics import accuracy_score
print("The Accuracy3 Score is: " , accuracy_score(y_test, y_pred_new))
## NMI
from sklearn.metrics.cluster import normalized_mutual_info_score
print("The NMI3 is: " , normalized_mutual_info_score(y_test, y_pred_new, average_method='arithmetic'))



## Accuracy
from sklearn.metrics import accuracy_score
print("The Accuracy Score is: " , accuracy_score(y_test, y_pred))

## NMI
from sklearn.metrics.cluster import normalized_mutual_info_score
print("The NMI is: " , normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic'))

## MSE New
from scipy.spatial.distance import euclidean
summation = 0
for i in range(len(max_index_col)):
    difference = euclidean( np.reshape(boc[i],(1, 400)) , out_centroid[max_index_col[i]])
    squared_difference = difference**2
    summation = summation + squared_difference
SSE = summation
MSE = SSE/len(max_index_col)
print("The Sum Squared Error is: " , SSE)
print("The Mean Squared Error is: " , MSE)

##########################################################
##print("--- %s seconds ---" % (time.time() - start_time))
