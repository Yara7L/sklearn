from sklearn.decomposition import PCA,MiniBatchSparsePCA,SparsePCA

# sparsePCA,MiniBatchSparsePCA用了L1正则，消除了非主要成分的影响。MiniBatchSparsePCA降维，
# 一部分样本特征与迭代，可能会降低精度

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn .datasets.samples_generator import make_blobs

X,y=make_blobs(n_samples=10000,n_features=3,centers=[[3,3,3],[0,0,0],[1,1,1],[2,2,2]],
                cluster_std=[0.2,0.1,0.2,0.2],random_state=9)
fig=plt.figure()
ax=Axes3D(fig,rect=[0,0,1,1],elev=30,azim=20)
plt.scatter(X[:,0],X[:,1],X[:,2],marker='o')
plt.show()

pca=PCA(n_components=3)
pca.fit(X)
print("PCA------n_component=3---------")
print(pca.explained_variance_ratio_,pca.explained_variance_)

pca=PCA(n_components=2)
pca.fit(X)
print("PCA------n_component=2---------")
print(pca.explained_variance_ratio_,pca.explained_variance_)

X_new=pca.transform(X)
plt.scatter(X_new[:,0],X_new[:,1],marker='o')
plt.show()

pca=PCA(n_components=0.95)
pca.fit(X)
print("PCA------n_component=0.95---------")
print(pca.explained_variance_ratio_,pca.explained_variance_)

pca=PCA(n_components=0.99)
pca.fit(X)
print("PCA------n_component=0.99---------")
print(pca.explained_variance_ratio_,pca.explained_variance_)

# pca=PCA(n_components='mle')
# pca.fit(X)
# print("PCA------n_component='mle'--------")
# print(pca.explained_variance_ratio_,pca.explained_variance_)