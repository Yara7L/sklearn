from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

# 多分类问题
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
print("multiclass================")
print(classif.fit(X, y).predict([[1,2]]))

y=LabelBinarizer().fit_transform(y)
classif.fit(X,y).predict(X)
print("labelbinarizer================")
print(classif.fit(X, y).predict([[1,2]]))

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y=MultiLabelBinarizer().fit_transform(y)
print("mutilabelbinarizer================")
print(classif.fit(X, y).predict([[1,2]]))
