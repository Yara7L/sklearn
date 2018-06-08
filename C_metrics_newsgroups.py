
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism','soc.religion.christian','comp.graphics', 'sci.med']
twenty_train=fetch_20newsgroups(subset="train",categories=categories,shuffle=True,random_state=42)

print(twenty_train.target_names)

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(twenty_train.data)
print("训练数据共{0}篇，词汇计数为{1}个".format(X_train_counts.shape[0],X_train_counts.shape[1]))

count=count_vect.vocabulary_.get(u'algorithm')
print("algorithm的出现次数为{0}".format(count))

from sklearn.feature_extraction.text import TfidfTransformer
'''
tf_transformer=TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf=tf_transformer.transform(X_train_counts)
'''
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(X_train_tfidf,twenty_train.target)
print("分类器的相关信息")
print(clf)

docs_new=['this GPU is great and fast','God is love']
X_new_counts=count_vect.transform(docs_new)
X_new_tfidf=tfidf_transformer.transform(X_new_counts)

predicted=clf.predict(X_new_tfidf)

for doc,category in zip(docs_new,predicted):
    print('%r=>%s' % (doc,twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline
text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB()),])
text_clf=text_clf.fit(twenty_train.data,twenty_train.target)
print(text_clf)

import numpy as np 
twenty_test=fetch_20newsgroups(subset="test",categories=categories,shuffle=True,random_state=42)
'''
twenty_test=load_files('20news-bydate/20news-bydate-test',
                        categories=categories,
                        load_content = True, 
                        encoding='latin1',
                        decode_error='strict',
'''
docs_test=twenty_test.data
predicted=text_clf.predict(docs_test)
print("准确率为：")
print(np.mean(predicted==twenty_test.target))

from sklearn.linear_model import SGDClassifier
text_clf=Pipeline([('vect',CountVectorizer()),
                   ('tfidf',TfidfTransformer()),
                   ('clf',SGDClassifier(
                       loss='hinge',
                       penalty='l2',
                       alpha=0.001,
                       n_iter=5,
                       random_state=42)),
                  ])
_=text_clf.fit(twenty_train.data,twenty_train.target)
predicted=text_clf.predict(docs_test)
print("准确率：")
print(np.mean(predicted==twenty_test.target))

from sklearn import metrics
print("打印分类性能指标：")
print(metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
print("打印混淆矩阵：")
print(metrics.confusion_matrix(twenty_test.target,predicted))

from sklearn.model_selection import GridSearchCV
parameters={'vect__ngram_range':[(1,1),(1,2)],
            'tfidf__use_idf':(True,False),
            'clf__alpha':(0.01,0.001)}
gs_clf=GridSearchCV(text_clf,parameters,n_jobs=1)
print(gs_clf)

gs_clf=gs_clf.fit(twenty_train.data[:400],twenty_train.target[:400])
print(twenty_train.target_names[gs_clf.predict(['An apple a day keeps the doctor away'])[0]])
print("最佳准确率：%r" % (gs_clf.best_score_))
for param_name in sorted(parameters.keys()):
    print("%s:%r" % (param_name,gs_clf.best_params_[param_name]))
