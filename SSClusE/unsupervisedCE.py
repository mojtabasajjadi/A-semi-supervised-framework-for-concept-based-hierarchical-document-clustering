# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Preprocess ,
#   Creating bag of concept for documents,


############################## Import librerie's
import boc

############################## Construct bag of concept for documents
a=boc.BOCModel(doc_path="dataset/Reuters4class.txt")
mtx,w2c,idx=a.fit(save_path="save/")

print(mtx.shape)

