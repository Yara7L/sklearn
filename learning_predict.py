import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.svm import SVC  

X = np.array([[-1,-1],[-2,-1],[1,1],[2,1],[-1,1],[-1,2],[1,-1],[1,-2]])  
y = np.array([0,0,1,1,2,2,3,3]) 
clf = SVC(probability=True)  
clf.fit(X, y)  
decision_function_shape="ovo"
print(clf.decision_function(X)) 
'''  
decision_function is to predict confidence scores for samples.
The confidence score for a sample is the signed distance of that sample to the hyperplane.
decision_function_shape="ovr"时是4个值，为ovo时是6个值。  
''' 
print('============')
print(clf.predict_proba(X))
print(clf.predict(X))  
 #这个是得分,每个分类器的得分，取最大得分对应的类。  

plot_step=0.02  
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),  
                     np.arange(y_min, y_max, plot_step))  
  
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #对坐标风格上的点进行预测，来画分界面。其实最终看到的类的分界线就是分界面的边界线。  
Z = Z.reshape(xx.shape)  
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)  
plt.axis("tight")  
  
class_names="ABCD"  
plot_colors="rybg"  
for i, n, c in zip(range(4), class_names, plot_colors):  
    idx = np.where(y == i) #i为0或者1，两个类  
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, label="Class %s" % n)  
plt.xlim(x_min, x_max)  
plt.ylim(y_min, y_max)  
plt.legend(loc='upper right')  
plt.xlabel('x')  
plt.ylabel('y')  
plt.title('Decision Boundary')  
plt.show()  