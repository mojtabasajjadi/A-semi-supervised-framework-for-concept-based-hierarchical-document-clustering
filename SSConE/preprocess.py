# Author: Seyed Mojtaba Sadjadi
# -*- coding: utf-8 -*-

# in this file we performe:
#   Preprocess Phase,
#   Creating word vector's for unlabeled data,
#   Extracting important keywords.



############################## Import librerie's
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
from more_itertools import sort_together
import itertools as it
from sklearn.model_selection import train_test_split

############################## Methods used in procedure
def getImportantVocab(n_vocab):
    v = []
    l = []
    temp = tf_idf_vectors.argmax(0)
    for i in range(len(temp)):
        v.append(features[i])
        l.append(y_train_labeled[temp[i]])
    return v , l  


############################## Read Data
docs = []
f = open('dataset/Reuters4class.txt')
line = f.readline()
while line:
    line = line.lower()
    line = re.sub('[^a-zA-Z]', ' ', line )
    line = re.sub(r'\s+', ' ', line)
    line = remove_stopwords(line)
    docs.append(line)
    line = f.readline()
f.close()

############################## Read labels of document from text file
document_labels = []
with open('dataset/Reuters4class_label.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line
        # add item to the list
        document_labels.append(current_place)  
filehandle.close


############################## Split data into train and test , labeled and unlabeled
X_train, X_test, y_train, y_test = train_test_split(docs, document_labels, test_size = 0.6, random_state = 0)
X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(X_train, y_train, train_size  = 200, random_state = 0)

for i in range(len(y_train_unlabeled)):
    y_train_unlabeled[i] = "unlabeled"



############################## Preprocessing on X_train_labeled Data with TF-IDF (Extract important keywords)
tfidf = TfidfVectorizer(
    analyzer='word',
    min_df = 5,
    max_df = 0.85,
#    max_features = 200,
#    stop_words = 'english',
#    lowercase = True
)
tfidf.fit_transform(X_train_labeled)
text = tfidf.transform(X_train_labeled)
tf_idf_vectors = text.toarray()
features = tfidf.get_feature_names()

vocab_labeled , label_labeled  = getImportantVocab(3)




############################## tokenize the X_train_unlabeled
X_train_unlabeled2 = ' '.join(X_train_unlabeled)
import nltk
# to convert our article into sentences
X_train_unlabeled_all_sentences = nltk.sent_tokenize(X_train_unlabeled2) 
# to convert sentences into words
X_train_unlabeled_all_words = [nltk.word_tokenize(sent) for sent in X_train_unlabeled_all_sentences]

word_list = X_train_unlabeled_all_words[0]
vocab_unlabeled = list(dict.fromkeys(word_list))
vocab_unlabeled = [x for x in vocab_unlabeled if x not in vocab_labeled]


label_unlabeled = []
for i in range(len(vocab_unlabeled)):
    label_unlabeled.append("unlabeled")

    

############################## Creating Word2Vec Model and vector's
def tokenize(doc_path):
    with open(doc_path, "r",encoding='utf-8') as f:
        for doc in f:
            temp = doc
            temp = temp.lower()
            temp = re.sub('[^a-zA-Z]', ' ', temp )
            temp = re.sub(r'\s+', ' ', temp)
            temp = remove_stopwords(temp)
            temp = temp.rstrip().split(" ")
            temp = [x for x in temp if x]
            yield temp
tokenized_docs=tokenize('dataset/Reuters4class.txt')

vocabs = [vocab_labeled + vocab_unlabeled]
from gensim.models import Word2Vec
embedding_dim = 400
model = Word2Vec( size=embedding_dim, window=5, min_count=1, workers=4)
model.build_vocab(vocabs)
model.train(tokenized_docs, total_examples=model.corpus_count, epochs=5)





w2v_vector_labeled = []
for i in range(len(vocab_labeled)):
    w2v_vector_labeled.append(np.reshape( model.wv[vocab_labeled[i]] , (1,400)))
w2v_vector_unlabeled = []
for j in range(len(vocab_unlabeled)):
    w2v_vector_unlabeled.append(np.reshape( model.wv[vocab_unlabeled[j]] , (1,400)))

idx2word = model.wv.index2word


#############################  Save splited data's into file 

############################## TRAIN DATA 
filename_xl = "x_l.pkl" #labeled documents
filename_xu = "x_u.pkl" #unlabeled documents
filename_yl = "y_l.pkl" #labels of labeled documents
filename_yu = "y_u.pkl" #labels of unlabeled documents = "unlabeled"
filename_vl = "v_l.pkl" #vocabularies of labeled documents
filename_vu = "v_u.pkl" #vocabularies of unlabeled documents

open_file = open(filename_xl, "wb")
pickle.dump(w2v_vector_labeled, open_file)
open_file.close()
open_file = open(filename_xu, "wb")
pickle.dump(w2v_vector_unlabeled, open_file)
open_file.close()

open_file = open(filename_yl, "wb")
pickle.dump(label_labeled, open_file)
open_file.close()
open_file = open(filename_yu, "wb")
pickle.dump(label_unlabeled, open_file)
open_file.close()

open_file = open(filename_vl, "wb")
pickle.dump(vocab_labeled, open_file)
open_file.close()
open_file = open(filename_vu, "wb")
pickle.dump(vocab_unlabeled, open_file)
open_file.close()

############################## TEST DATA 
filename_x_test = "x_test.pkl"
filename_y_test = "y_test.pkl"

open_file = open(filename_x_test, "wb")
pickle.dump(X_test, open_file)
open_file.close()
open_file = open(filename_y_test, "wb")
pickle.dump(y_test, open_file)
open_file.close()
    
    