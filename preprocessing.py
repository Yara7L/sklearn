from sklearn import preprocessing
import numpy as np 

X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])

def standarization():
    X_train = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
    X_scaled = preprocessing.scale(X_train)
    print(X_scaled)
    print(X_scaled.mean(axis=0),X_scaled.std(axis=0))

    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler.mean_,scaler.scale_)
    print(scaler.transform(X_train))
    # the scaler instance can be then used on new data to transform it the same way it did on the training set


    min_max_scaler = preprocessing.MinMaxScaler()
    print(min_max_scaler.fit_transform(X_train))

    # scaling sparse data => MaxAbsScalr,maxabs_scale  recieve the scipy.sparse
    # scaling data with outliers => RobustScaler,robust_scale
    # centering kernel matrics => KernerlCenterer

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def non_linear_transformation():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    X_train_trans = quantile_transformer.fit_transform(X_train)
    X_test_trans = quantile_transformer.transform(X_test)
    print(np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
    print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))#量纲，cm，百分位数，值小
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    X_trans = quantile_transformer.fit_transform(X)
    print(quantile_transformer.quantiles_)

def normalization():
    # 
    X_normalized = preprocessing.normalize(X, norm='l2')
    print(X_normalized)
    normalizer=preprocessing.Normalizer().fit(X)
    print(normalizer)
    print(normalizer.transform(X))

def binarization():
    # thresholding numerical features to get boolean values
    binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
    print(binarizer)
    print(binarizer.transform(X))
    binarizer_threshold=preprocessing.Binarizer(threshold=1.1)
    print(binarizer_threshold.transform(X))

def encoding_categorical_features():
    enc = preprocessing.OneHotEncoder()
    # m个,m**2
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
# OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
#    handle_unknown='error', n_values='auto', sparse=True)      
    print(enc.transform([[0, 1, 3]]).toarray())

    enc = preprocessing.OneHotEncoder(n_values=[2, 3, 4])
    enc.fit([[1, 2, 3], [0, 2, 0]])  
# OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
#        handle_unknown='error', n_values=[2, 3, 4], sparse=True)
    print(enc.transform([[1, 0, 0]]).toarray())

from sklearn.preprocessing import Imputer
import scipy.sparse as sp
def imputation_missing():
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit([[1, 2], [np.nan, 3], [7, 6]])
    X = [[np.nan, 2], [6, np.nan], [7, 6]]
    print(imp.transform(X)) 

    X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
    imp = Imputer(missing_values=0, strategy='mean', axis=0)
    imp.fit(X)
    X_test = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
    print(imp.transform(X_test))

from sklearn.preprocessing import PolynomialFeatures
def polynomial_features():
    X = np.arange(6).reshape(3, 2)
    print(X)                                                 
    poly = PolynomialFeatures(2)
    print(poly.fit_transform(X))
    #the features of X have been transformaed{X1,X2}to{1,X1,X2,X1**2,X1X2,X2**2} 
    X = np.arange(9).reshape(3, 3)
    print(X)
    poly = PolynomialFeatures(degree=3, interaction_only=True)
    print(poly.fit_transform(X))
    #the features of X have been transformaed{X1,X2,X3}to{1,X1,X2,X3,X1X2,X1X3,X2X3,X1X2X3} 

from sklearn.preprocessing import FunctionTransformer
def custom_transformers():
    transformer=FunctionTransformer(np.log1p)
    X = np.array([[0, 1], [2, 3]])
    print(transformer.transform(X))

if __name__=="__main__":
    # standarization()
    # non_linear_transformation()
    # normalization()
    # binarization()
    # encoding_categorical_features()
    # imputation_missing()
    # polynomial_features()
    custom_transformers()

