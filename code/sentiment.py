# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
import nltk
from sklearn import metrics
from Tools.scripts.treesync import raw_input
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from numpy import *
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import collections
import numpy as np


# 接着在模型的compile中设置metrics

def plotPR(yt, ys, title=None):
    '''
    绘制precision-recall曲线
    :param yt: y真值
    :param ys: y预测值, recall,
    '''

    from sklearn import metrics
    from matplotlib import pyplot as plt1
    precision, recall, thresholds = metrics.precision_recall_curve(yt, ys)

    plt1.plot(precision, recall, 'darkorange', lw=1, label='x=precision')
    plt1.plot(recall, precision, 'blue', lw=1, label='x=recall')
    plt1.legend(loc='best')
    plt1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt1.title('Precision-Recall curve %s for sentiment-analysis'%title)
    plt1.ylabel('Recall')
    plt1.xlabel('Precision')
    plt1.show()

def plotRUC(yt, ys, title=None):
    '''
    绘制ROC-AUC曲线
    :param yt: y真值
    :param ys: y预测值
    '''
    from sklearn import metrics
    from matplotlib import pyplot as plt2
    f_pos, t_pos, thresh = metrics.roc_curve(yt, ys)
    auc_area = metrics.auc(f_pos, t_pos)
    print('auc_area: {}'.format(auc_area))

    plt2.plot(f_pos, t_pos, 'darkorange', lw=2, label='AUC = %.2f' % auc_area)
    plt2.legend(loc='lower right')
    plt2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt2.title('ROC-AUC curve for %s sentiment-analysis'%title )
    plt2.ylabel('True Pos Rate')
    plt2.xlabel('False Pos Rate')
    plt2.show()

def giniCoefficient(x, y):
    '''
    gini系数计算
      :param x: 推测值（人口）
      :param y: 实际值（财富）
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    x.__add__(0)
    y.__add__(0)

    x = np.cumsum(x)
    if x[-1] != 0:
        x = x / x[-1]

    y = np.cumsum(y)
    if y[-1] != 0:
        y = y / y[-1]

    area = metrics.auc(x, y, reorder=True)
    gini_cof = 1 - 2 * area

    return gini_cof if math.fabs(gini_cof) > pow(math.e, -6) else 0

if __name__ == '__main__':
    ## EDA
    maxlen = 0
    #句子最大长度
    word_freqs = collections.Counter()
    #词频
    num_recs = 0
    #样本数
    with open('./LSTM_data/train_data.txt', 'r+', encoding='UTF-8') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words = nltk.word_tokenize(sentence.lower())
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
    print('max_len ', maxlen)
    print('nb_words ', len(word_freqs))
    #不同单词的个数
    ## 准备数据
    MAX_FEATURES = 2000
    MAX_SENTENCE_LENGTH = 40
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    #一个是填充用的0，一个是伪单词UNK
    word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    X = np.empty(num_recs, dtype=list)
    y = np.zeros(num_recs)
    i = 0
    #两个lookuptables用于单词数字转换
    with open('LSTM_data/train_data.txt', 'r+', encoding='UTF-8') as f:
        for line in f:
            label, sentence = line.strip().split("\t")
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            y[i] = int(label)
            i += 1
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

    ## 数据划分
    #80%作为训练数据 20%作为测试数据
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    # 整个数据轮流十次，梯度下降为32

    ## 网络构建
    EMBEDDING_SIZE = 128
    HIDDEN_LAYER_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # 这里损失函数用 binary_crossentropy， 优化方法用 adam。
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    ## 网络训练
    model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

    ## 预测
    print('{}   {}      {}'.format('预测', '真实', '句子'))
    for i in range(5):
        idx = np.random.randint(len(Xtest))
        xtest = Xtest[idx].reshape(1, 40)
        ylabel = ytest[idx]
        ypred = model.predict(xtest)[0][0]
        sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
        print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

    ##### 自己输入
    INPUT_SENTENCES = ['I like reading.', 'You are so boring.']
    XX = np.empty(len(INPUT_SENTENCES), dtype=list)
    i = 0
    for sentence in INPUT_SENTENCES:
        words = nltk.word_tokenize(sentence.lower())
        seq = []
        for word in words:
            if word in word2index:
                seq.append(word2index[word])
            else:
                seq.append(word2index['UNK'])
        XX[i] = seq
        i += 1

    XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    labels = [int(round(x[0])) for x in model.predict(XX) ]
    label2word = {1: '积极', 0: '消极'}
    for i in range(len(INPUT_SENTENCES)):
        print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))
    ##用户自定义输入
    sentence1 = raw_input("Enter your sentence1:")
    sentence2 = raw_input("Enter your sentence2:")
    i = 0
    SENTENCE = [sentence1, sentence2]
    XXX = np.empty(len(SENTENCE), dtype=list)
    for sentences in SENTENCE:
        words1 = nltk.word_tokenize(sentences.lower())
        seq1 = []
        for word1 in words1:
            if word1 in word2index:
                seq1.append(word2index[word1])
            else:
                seq1.append(word2index['UNK'])
        XXX[i] = seq1
        i += 1

    XXX = sequence.pad_sequences(XXX, maxlen=MAX_SENTENCE_LENGTH)
    labels1 = [int(round(x[0])) for x in model.predict(XXX) ]
    label2word = {1: '积极', 0: '消极'}
    for i in range(len(SENTENCE)):
        print('{}   {}'.format(label2word[labels1[i]], SENTENCE[i]))

    ##神经网络模型评估
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("\n***************神经网络模型评估******************")
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    xxtest = [int(round(x[0])) for x in model.predict(Xtest)]
    #precisions = cross_val_score(model, xxtest, ytest, scoring='precision')
    #print(u'精确率：', np.mean(precisions), precisions)
    #recalls = cross_val_score(model, xxtest, ytest, scoring='recall')
    #print(u'召回率：', np.mean(recalls), recalls)
    #plt.scatter(recalls, precisions)
    #fls = cross_val_score(model, xxtest, ytest, scoring='f1')
    print('**************混淆矩阵************')
    print(metrics.confusion_matrix(ytest, xxtest))
    #print('综合指标评价', np.mean(fls), fls)
    print('*********sklearn评估报告***************')
    print(classification_report(xxtest, ytest, target_names=['消极', '积极']))
    string1 = 'LSTM'
    plotPR(ytest, xxtest,string1)
    plotRUC(ytest, xxtest,string1)
    print('*********gini指数***************')
    print(giniCoefficient(ytest, xxtest))


    ##逻辑回归模型评估
    string2='Logistic Regression'
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print("\n***************逻辑回归分类器模型评估******************")
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    classifier = LogisticRegression()
    classifier.fit(Xtest, ytest)
    precisions = cross_val_score(classifier, Xtest, ytest, cv=10, scoring='precision')
    print(u'精确率：', np.mean(precisions), precisions)
    recalls = cross_val_score(classifier, Xtest, ytest, cv=10, scoring='recall')
    print(u'召回率：', np.mean(recalls), recalls)
    plt.scatter(recalls, precisions)
    fls = cross_val_score(classifier, Xtest, ytest, cv=10, scoring='f1')
    print('**************混淆矩阵************')
    print(metrics.confusion_matrix(ytest, classifier.predict(Xtest)))
    print('综合指标评价', np.mean(fls), fls)
    print('*********sklearn评估报告***************')
    print(classification_report(classifier.predict(Xtest), ytest, target_names=['消极', '积极']))
    plotPR(ytest, classifier.predict(Xtest),string2)
    plotRUC(ytest, classifier.predict(Xtest),string2)
    print('*********gini指数***************')
    print(giniCoefficient(ytest, classifier.predict(Xtest)))


