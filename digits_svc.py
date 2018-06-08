from sklearn import datasets,svm,metrics
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.grid_search import GridSearchCV

# 手写数字分类，主要实践了Kfold,cross_val_score,GridSearchCV的模型选择的应用

def data_figure(digits):
    '''
    展示数据
    '''
    images_and_labels=list(zip(digits.images,digits.target))
    for index,(image,label) in enumerate(images_and_labels[:10]):
        plt.subplot(2,10,index+1)
        plt.axis('off')
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title('%i' % label)
    plt.show()

def svc_rbf(digits):
    '''
    SVC(rbf)
    '''
    clf=svm.SVC(gamma=0.001,C=100.)
    # clf.fit(digits.data[:-1],digits.target[:-1])
    #2D->1D  问题
    # test = [0, 0, 10, 14, 8, 1, 0, 0,
    #         0, 2, 16, 14, 6, 1, 0, 0,
    #         0, 0, 15, 15, 8, 15, 0, 0,
    #         0, 0, 5, 16, 16, 10, 0, 0,
    #         0, 0, 12, 15, 15, 12, 0, 0,
    #         0, 4, 16, 6, 4, 16, 6, 0,
    #         0, 8, 16, 10, 8, 16, 8, 0,
    #         0, 1, 8, 12, 14, 12, 1, 0]
    # print("对图片的预测结果为：")
    # print(clf.predict(np.asarray(test)))
    clf.fit(digits.data[:1000],digits.target[:1000])
    expected=digits.target[1000:]
    predicted=clf.predict(digits.data[1000:])
    print("分类器预测结果评估：\n%s\n" % (metrics.classification_report(expected,predicted)))

def svc_linear(digits):
    '''
    SVC(linear)
    3折交叉验证
    '''
    X_digits=digits.data
    y_digits=digits.target
    svc=svm.SVC(C=1.0,kernel='linear')
    score_svc=svc.fit(X_digits[:-100],y_digits[:-100]).score(X_digits[-100:],y_digits[-100:])
    print(score_svc)

    # 3折交叉验证
    X_folds=np.array_split(X_digits,3)
    y_folds=np.array_split(y_digits,3)
    scores_cross=list()

    for k in range(3):
        X_train=list(X_folds)
        X_test=X_train.pop(k)
        X_train=np.concatenate(X_train)

        y_train=list(y_folds)
        y_test=y_train.pop(k)
        y_train=np.concatenate(y_train)

        scores_cross.append(svc.fit(X_train,y_train).score(X_test,y_test))
    #mean
    print(scores_cross)
    print(np.mean(scores_cross))

    # scores_kf = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in k_fold]
    # print(scores_kf)

def k_fold_indices():
    '''
    k_fold_indices拆分小例子
    '''
    X = ["a", "a", "b", "c", "c", "c"]
    k_fold = KFold(n=6, n_folds=3)
    for train_indices, test_indices in k_fold:
        print('Train: %s | test: %s' % (train_indices, test_indices))

def svc_cross_validation(digits):
    '''
    cross_validation
    '''
    X = digits.data
    y = digits.target

    svc = svm.SVC(kernel='linear')
    C_s = np.logspace(-10, 0, 10)

    scores = list()
    scores_std = list()
    for C in C_s:
        svc.C = C
        this_scores = cross_val_score(svc, X, y, n_jobs=1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))

    plt.figure(1, figsize=(4, 3))
    plt.clf() #close figure,  cla() close axis,  close() close a figure window
    plt.semilogx(C_s, scores)
    plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)
    plt.show()

def svc_grid(digits):
    '''GridSearchCV'''
    X_digits=digits.data
    y_digits=digits.target
    Cs=np.logspace(-6,-1,10)
    clf=GridSearchCV(estimator=svm.SVC(kernel='linear'),param_grid=dict(C=Cs),n_jobs=-1)
    clf.fit(X_digits[:1000],y_digits[:1000])
    score_best=clf.best_score_
    print(score_best)
    C_best=clf.best_estimator_.C
    print(C_best)
    score_train=clf.score(X_digits[1000:],y_digits[1000:])
    print(score_train)

if __name__=='__main__':
    digits=datasets.load_digits()
    #8*8的数字点阵图，代表颜色深度
    # print(digits.data.shape)
    # print(digits.target)
    k_fold_indices()
    data_figure(digits)
    svc_rbf(digits)
    svc_linear(digits)
    svc_cross_validation(digits)
    svc_grid(digits)
