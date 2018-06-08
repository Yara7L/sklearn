import scipy as sp 
import numpy as np 
from sklearn import cluster
import matplotlib.pyplot as plt 


try:
    face=sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face=misc.face(gray=True)

X=face.reshape((-1,1))
k_means=cluster.KMeans(n_clusters=5,n_init=1)
k_means.fit(X)

values=k_means.cluster_centers_.squeeze()
labels=k_means.labels_

face_compressed=np.choose(labels,values)
face_compressed.shape=face.shape

plt.imshow(face_compressed)
plt.show()

face=sp.misc.imresize(face,0.10)/255
