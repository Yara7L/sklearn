from sklearn import datasets
import numpy as np 
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.linear_model import LassoCV,Lasso
from sklearn import linear_model
import matplotlib.pyplot as plt

diabetes=datasets.load_diabetes()
diabetes_X_train=diabetes.data[:-20]
diabetes_y_train=diabetes.data[:-20]
diabetes_X_test=diabetes.data[-20:]
diabetes_y_test=diabetes.data[-20:]

regr=linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
print(regr.coef_)
regr_loss=np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
print(regr_loss)
score=regr.score(diabetes_X_test,diabetes_y_test)
print(score)


X=diabetes.data[:150]
y=diabetes.target[:150]

lasso = Lasso(random_state=0)
alphas=np.logspace(-4,-0.5,30)

scores=list()
scores_std=list()

n_folds=3

for alpha in alphas:
    lasso.alpha=alpha
    this_scores=cross_val_score(lasso,X,y,cv=n_folds,n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    
scores,scores_std=np.array(scores),np.array(scores_std)

plt.figure().set_size_inches(8,6)
plt.semilogx(alphas,scores)

std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

plt.show()