# -*- coding: utf-8 -*-
from os import chdir
chdir('D:/kantor.py/')
list_of_strings = ['working', 'with', 'text']
for s in list_of_strings:
    print(s)
import numpy as np  # a conventional alias
from sklearn.feature_extraction.text import CountVectorizer
filenames = ['data/austen-brontë/Austen_Emma.txt',           
             'data/austen-brontë/Austen_Pride.txt',          
             'data/austen-brontë/Austen_Sense.txt',          
             'data/austen-brontë/CBronte_Jane.txt',          
             'data/austen-brontë/CBronte_Professor.txt',     
             'data/austen-brontë/CBronte_Villette.txt']
vectorizer = CountVectorizer(input='filename')               
dtm = vectorizer.fit_transform(filenames)  # a sparse matrix 
vocab_list = vectorizer.get_feature_names()
# for reference, note the current class of `dtm`  
type(dtm)                                         
dtm = dtm.toarray()  # convert to a regular array 
vocab = np.array(vocab_list)
# the first file, indexed by 0 in Python, is *Emma*                 
filenames[0] == 'data/austen-brontë/Austen_Emma.txt'                
                                                                    
# use the standard Python list method index(...)                    
house_idx = vocab_list.index('house')                               
dtm[0, house_idx]                                                   
                                                                    
# alternatively, use NumPy indexing                                 
# in R this would be essentially the same, dtm[1, vocab == 'house'] 
dtm[0, vocab == 'house']  

print(vocab[house_idx])

print(dtm.shape)
for fn in filenames:
    pass#print(fn)
    
print(len(vocab))
print(vocab[500:550])  # look at some of the vocabulary
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(dtm)
print(np.round(dist, 2))
print(dist[1, 3])
print(dist[1, 3] > dist[3, 5])
norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
dtm_normed = dtm / norms
similarities = np.dot(dtm_normed, dtm_normed.T)
print(np.round(similarities, 2))
print(similarities[1, 3])
import os  # for os.path.basename
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

# short versions of filenames:
# convert 'data/austen-brontë/Austen_Emma.txt' to 'Austen_Emma'
names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]

# color-blind-friendly palette
for x, y, name in zip(xs, ys, names):
    color = 'orange' if "Austen" in name else 'skyblue'
    plt.scatter(x, y, c=color)
    plt.text(x, y, name)

plt.show()
# après Jeremy M. Stober, Tim Vieira
# https://github.com/timvieira/viz/blob/master/mds.py
mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
    ax.text(x, y, z, s)
plt.show()
from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix = ward(dist)

# match dendrogram to that returned by R's hclust()
print(dendrogram(linkage_matrix, orientation="right", labels=names))

#plt.tight_layout()  # fixes margins