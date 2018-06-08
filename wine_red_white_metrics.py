from operator import itemgetter
import sys
import pandas
import numpy 
import matplotlib.pyplot as pyplot
import seaborn as sns  # seaborn 数据可视化
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,cohen_kappa_score

# 红酒白酒分类，熟悉metrics中性能评价

def load_data():
    red=pandas.read_csv("E:/ML/dataset/winequality-red.csv",sep=';')
    white =pandas.read_csv("E:/ML/dataset/winequality-white.csv", sep=';')
    red['type']=1
    white['type']=0
    wines=red.append(white,ignore_index=True)

    #指定特征变量列
    X=wines.ix[:,0:11]
    #指定标签列。展平多维数组
    y=numpy.ravel(wines.type)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
    # 标准化数据
    scaler=StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    return red,white,wines,X_train,X_test,y_train,y_test

def figure_show():
    red,white,wines,X_train,X_test,y_train,y_test=load_data()
    fig,ax=pyplot.subplots(1,2)
    ax[0].hist(red.alcohol,10,facecolor='red',alpha=0.5,label='Red wine')
    ax[1].hist(white.alcohol,10,facecolor='white',ec='black',lw=0.5,alpha=0.5,label='White wine')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=0.5,hspace=0.05,wspace=1)
    ax[0].set_ylim([0,1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")
    fig.suptitle("Distribution of Alcohol in % Vol")
    pyplot.show()

    corr=wines.corr()
    sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
    pyplot.show()


def built_model():
    model=Sequential()
    model.add(Dense(16,activation='relu',input_shape=(11,)))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

def train():

    red,white,wines,X_train,X_test,y_train,y_test=load_data()
    model=built_model()
    print("============================================")
    print(model.output_shape)
    print("============================================")
    print(model.summary())
    print("============================================")    
    # print(model.get_config())
    # print("============================================")
    # print(model.get_weights())
    # print("============================================")
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=3,batch_size=1,verbose=1)

    y_pred=model.predict(X_test)
    y_pred = (y_pred > 0.5) # continue变为discrete，Boolean matrix
    score=model.evaluate(X_test,y_test,verbose=1)
    print(score)

    # y_test=numpy.asarray(y_test)
    # y_test=y_test.reshape(2145,1)
    # print(y_test.shape)

    print(confusion_matrix(y_test,y_pred))
    print(precision_score(y_test,y_pred))
    print(recall_score(y_test,y_pred))
    print(f1_score(y_test,y_pred))
    print(cohen_kappa_score(y_test,y_pred))

if __name__=='__main__':
    figure_show()
    train()