from sklearn import preprocessing 

# one-hot编码，又称为一位有效编码。
# 首先将分类映射到整数值，然后每个整数值被表示为二进制向量，
# 除了整数的索引（１）之外，其他都为０
encode=preprocessing.OneHotEncoder()
encode.fit([[0,0,3],[1,1,0],[1,0,2]]) 
# 拟合
array=encode.transform([[0,1,3]]).toarray()
#转化 
print(array)


from numpy import argmax
# define input string
data = 'hello world'
print(data)
# define universe of possible input values
alphabet = 'abcdefghijklmnopqrstuvwxyz '
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# one hot encode
onehot_encoded = list()
for value in integer_encoded:
       letter = [0 for _ in range(len(alphabet))]
       letter[value] = 1
       onehot_encoded.append(letter)
print(onehot_encoded)
# invert encoding
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tag_list = ['青年 吃货 唱歌',  
            '少年 游戏 叛逆',  
            '少年 吃货 足球'] 
vectorizer=CountVectorizer()
# 将文本中的词语转换为词频矩阵
X=vectorizer.fit_transform(tag_list)
# 计算词语出现的次数

transformer=TfidfTransformer()
tfidf=transformer.fit_transform(X)
# 将词频矩阵x统计成TF-IDF值
print(tfidf.toarray())