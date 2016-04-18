# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
from os import chdir, mkdir, path
from random import randrange
import gensim
from gensim.models import *
chdir("D:/kantor.py/")
inp_path = "texts_train_10_full.txt"
lab_path = "labels_train_10_full.txt"
def prepare():
    labels = {}
    count = 0
    with open(lab_path, "r") as lf:
        for label in lf:
            labels.setdefault(label.strip(), []).append(count)
            count += 1
    files = [0 for i in range(count)]
    for label, inds in labels.items():
        if not path.exists(label): mkdir(label)
        outf = open(label+"/part1.txt", "w")
        for i in inds: files[i] = outf
    ind = 0
    with open(inp_path, "r") as f:
        for line in f:
            files[ind].write(line)
            ind += 1
def prepare_parts(part):
    for label in labels:
        with open(label+"/part1.txt", "r") as f:
            with open(label+"/part"+str(part)+".txt", "w") as outf:
                for line in f:
                    if randrange(part) != 0: continue
                    outf.write(line)
labels = ["Rest","Business","Entertainment","PrivateLife"]
vectorizer = TfidfVectorizer(norm='l2')#CountVectorizer()
k = len(labels)
part = 10
#prepare_parts(part)
docs, classes = [], []
counts = [0 for i in range(len(labels))]
fname = "/part"+str(part)+".txt"
for i in range(len(labels)):
    with open(labels[i]+fname, "r") as f:
        for line in f:
            docs.append(line)
            classes.append(i)
            counts[i] += 1
"""model = Word2Vec(size=100, window=3, min_count=3, workers=4, sg=0)
model.build_vocab(docs)
model.train(docs)
print(len(docs))
print(model["цена"])
exit()"""
X = vectorizer.fit_transform(docs)
model = KMeans(n_clusters=k, init='random', max_iter=100, n_init=3)
res = model.fit_predict(X)
table = [[0 for j in range(len(labels))] for i in range(k)]
for i in range(len(res)):
    table[res[i]][classes[i]] += 1
for i in range(len(labels)):
    for j in range(k):
        table[j][i] = int(100*table[j][i]/counts[i])
print(table)