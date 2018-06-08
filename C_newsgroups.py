from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups_vectorized

# newsgroup的分类，聚类处理
# 利用sklearn进行文本特征提取(多种方式)
#   1）HashingVectorizer
#   2）CountVectorizer+TfidfTransformer
#   3）TfidfVectorizer


def hash_vector():
    # FeatureHasher不执行除 Unicode 或 UTF-8 编码之外的任何其他预处理
    # 特征提取(讲文本和图像转化数值特征，与特征选择不一致)method1
    # 不能反转模型，2**n个features
    vectorizer = HashingVectorizer(
        stop_words='english', non_negative=True, n_features=10000)
    fea_train = vectorizer.fit_transform(newsgroup_train.data)
    fea_test = vectorizer.fit_transform(newsgroups_test.data)

    print('Size of fea_train:' + repr(fea_train.shape))
    print('Size of fea_train:' + repr(fea_test.shape))
    print('The average feature sparsity is {0:.3f}%'.format(
        fea_train.nnz / float(fea_train.shape[0] * fea_train.shape[1]) * 100))
    return fea_train, fea_test


def count_tfidf():
    # 特征提取method2
    count_vector = CountVectorizer(stop_words='english', max_df=0.5)
    counts_train = count_vector.fit_transform(newsgroup_train.data)
    print("the shape of train is " + repr(counts_train.shape))

    count_v2 = CountVectorizer(vocabulary=count_vector.vocabulary_)
    counts_test = count_v2.fit_transform(newsgroups_test.data)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
    tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)
    return tfidf_train, tfidf_test


def tfidf_vector():
    # 特征提取method3
    tv = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    tfidf_train_2 = tv.fit_transform(newsgroup_train.data)
    tv2 = TfidfVectorizer(vocabulary=tv.vocabulary_)
    tfidf_test_2 = tv2.fit_transform(newsgroups_test.data)
    print("the shape of train is " + repr(tfidf_train_2.shape))
    print("the shape of test is " + repr(tfidf_test_2.shape))
    analyze = tv.build_analyzer()
    tv.get_feature_names()
    return tfidf_train_2, tfidf_test_2


def all_vector():
    print("all_vector load:")
    t0 = time()
    # raw_data = fetch_20newsgroups(subset='train').data
    # data_size_mb = sum(len(s.encode('utf-8')) for s in raw_data) / 1e6

    tfidf_train_3 = fetch_20newsgroups_vectorized(subset='train')
    tfidf_test_3 = fetch_20newsgroups_vectorized(subset='test')
    
    duration = time() - t0
    print("done in %fs"%duration)
    # print("done in %fs at %0.3fMB/s" % (duration, data_size_mb / duration))

    print("the shape of train is " + repr(tfidf_train_3.data.shape))
    print("the shape of test is " + repr(tfidf_test_3.data.shape))
    return tfidf_train_3.data, tfidf_test_3.data

# precision_score(),recall_score(),f1_score(),fbeta_score(),precision_recall_fscore_support()等
# 二分类问题与多分类(micro,samples,macro,weighted,None)问题.
def calculate_result(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='micro')
    m_recall = metrics.recall_score(actual, pred, average='micro')
    print('predict info:')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    # print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred)))


if __name__ == '__main__':
    # categories = [
    #     'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    #     'comp.sys.mac.hardware', 'comp.windows.x'
    # ]

    # newsgroup_train = fetch_20newsgroups(subset='train', categories=categories)
    # newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    # print(list(newsgroup_train.target_names))

    newsgroup_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    fea_train, fea_test = all_vector()

    # print(type(fea_train))
    # print(fea_train)
    # print(type(newsgroup_train.target))
    # print(newsgroup_train.target)

    #create the Multinomial Naive Bayesian Classifier
    clf = MultinomialNB(alpha=0.01)
    clf.fit(fea_train, newsgroup_train.target)
    pred = clf.predict(fea_test)
    calculate_result(newsgroups_test.target, pred)

    knnclf = KNeighborsClassifier()  #default with k=5
    knnclf.fit(fea_train, newsgroup_train.target)
    pred = knnclf.predict(fea_test)
    calculate_result(newsgroups_test.target, pred)

    svclf = SVC(kernel='linear')  #default with 'rbf'
    svclf.fit(fea_train, newsgroup_train.target)
    pred = svclf.predict(fea_test)
    calculate_result(newsgroups_test.target, pred)

    pred = KMeans(n_clusters=5)
    pred.fit(fea_test)
    calculate_result(newsgroups_test.target, pred.labels_)
