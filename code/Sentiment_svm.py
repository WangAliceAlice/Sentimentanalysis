# -*- coding: utf-8 -*-

from imp import reload

from Tools.scripts.treesync import raw_input
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve, classification_report, roc_curve, auc
from sklearn.svm import SVC
import sys
import matplotlib.pyplot as plt
reload(sys)  
#sys.setdefaultencoding('utf8')

# 加载文件，导入数据,分词
def loadfile():
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    pos=pd.read_excel('data/pos.xls',header=None,index=None)

    cw = lambda x: list(jieba.cut(x))#定义分词函数
    pos['words'] = pos[0].apply(cw)#拼接词
    neg['words'] = neg[0].apply(cw)

    print(pos['words'])
    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))#拼接

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
    
    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train,x_test
 


#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
    
#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)#维度和字典截断
    imdb_w2v.build_vocab(x_train)
    '''
    xiugaiguo
    '''
    #Train the model over train_reviews (this may take several minutes)
    imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('svm_data/train_vecs.npy',train_vecs)
    np.save('lg_data/train_vecs.npy',train_vecs)
    print(train_vecs.shape)
    #Train word2vec on test tweets
    imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=imdb_w2v.iter)
    '''
    xiugaiguo
    '''
    imdb_w2v.save('svm_data/w2v_model.pkl')
    imdb_w2v.save('lg_data/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    np.save('lg_data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)



def get_data():
    train_vecs=np.load('svm_data/train_vecs.npy')
    y_train=np.load('svm_data/y_train.npy')
    test_vecs=np.load('svm_data/test_vecs.npy')
    y_test=np.load('svm_data/y_test.npy') 
    return train_vecs,y_train,test_vecs,y_test

#========Logistic Regression========#
def lg_train(train_vecs,y_train,test_vecs,y_test):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'lg_data/model.pkl')
    print(clf.score(test_vecs, y_test))
    String2 = 'word2vec+lg'
    return clf,String2

def Precision(clf,String):

    doc_class_predicted = clf.predict(test_vecs)
    print('使用%s方法'%String)
    print(np.mean(doc_class_predicted == y_test))  # 预测结果和真实标签
    # 准确率与召回率
    precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(test_vecs))
    answer = clf.predict_proba(test_vecs)[:, 1]
    report = answer > 0.5
    print('使用%s方法'%String)
    print(classification_report(y_test, report, target_names=['neg', 'pos']))
    print("--------------------")
    from sklearn.metrics import accuracy_score
    print('准确率: %.2f' % accuracy_score(y_test, doc_class_predicted))  ####
    pred_probas = clf.predict_proba(test_vecs)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("%s"%String)
    plt.legend(loc='lower right')
    plt.show()

##训练svm模型
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    '''
    probability =true 是我添加的，否则不能使用clf.predict_proba()
    '''
    clf=SVC(probability=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/model.pkl')
    print(clf.score(test_vecs,y_test))
    String1='word2vec+svm'
    return clf,String1

##得到待预测单个句子的词向量    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('svm_data/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs
    ##得到待预测单个句子的词向量

##对输入句子得到词向量
def get_predict_vecslg(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('lg_data/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs

####对单个句子进行情感判断    
def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('svm_data/model.pkl')
     
    result=clf.predict(words_vecs)
    
    if int(result[0])==1:
        print(string,' positive')
    else:
        print(string,' negative')


def lg_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecslg(words)
    clf = joblib.load('lg_data/model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


if __name__=='__main__':
    
    
    ##导入文件，处理保存为向量
    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train.npy,y_test.npy
    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
    clf1,String1 = svm_train(train_vecs,y_train,test_vecs,y_test)
    Precision(clf1,String1)#训练svm并保存模型
    clf2,String2 = lg_train(train_vecs, y_train, test_vecs, y_test)
    Precision(clf2,String2)
    '''
    xiugaiguo
    '''

##对输入句子情感进行判断

    string1='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    string2='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    string =[string1 , string2]

##用户自定义输入
    sentence1 = raw_input("Enter your sentence1:")
    sentence2 = raw_input("Enter your sentence2:")
    SENTENCE = [sentence1, sentence2]

##使用word2vec+svm
    print("使用word2vec+SVM进行预测")
    for i in string:
        svm_predict(i)
    for sentence in SENTENCE:
        svm_predict(sentence)

##使用word2vec+lg
    print("使用word2vec+LG进行预测")
    for i in string:
        lg_predict(i)
    for sentence in SENTENCE:
        lg_predict(sentence)
