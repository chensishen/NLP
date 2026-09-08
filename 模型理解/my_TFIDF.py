import re
import copy
import numpy as np


def t1():
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=False)
    D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    X = v.fit_transform(D)
    print(X)
    print(v.vocabulary_)
    print(v.feature_names_)
    print(v.inverse_transform(X))
    print(v.transform({'foo': 4, 'unseen_feature': 3}))

def t0():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(corpus)
    print(f"从训练数据中提取出来的单词列表:\n{vectorizer.get_feature_names_out()}\n")
    print(f"词袋法转换后的结果:\n{X.toarray()}\n")

    # Y = transformer.fit_transform(X)
    # print(Y.toarray().round(2))

    # vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    # X2 = vectorizer2.fit_transform(corpus)
    # print(vectorizer2.get_feature_names_out())
    # print(X2.toarray())

def t2():
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(f"从训练数据中提取出来的单词列表:\n{vectorizer.get_feature_names_out()}\n")
    print(f"TF-IDF转换后的结果:\n{X.toarray().round(2)}\n")
    print(X.shape)

def get_words(corpus):
    words_set = set()
    all_words_list = [[] for _ in range(len(corpus))]

    for index, i in enumerate(corpus):
        i = re.sub(r'[^\w\s]', ' ', str(i))
        for j in i.lower().split():
            all_words_list[index].append(j)
            words_set.add(j)

    words_list = sorted(words_set)

    return all_words_list,words_list

def my_ConvertSequence(corpus):
    all_words_list,words_list = get_words(corpus)

    word_dict = dict(zip(words_list, range(len(words_list))))

    words_sequence = [[] for _ in range(len(corpus))]

    for index, i in enumerate(all_words_list):
        for j in i:
            words_sequence[index].append(word_dict[j])


    # print(words_list)
    # print(words_sequence)
    return words_list,words_sequence


def my_ConvertOnehot(corpus):
    all_words_list,words_list = get_words(corpus)

    word_dict = dict(zip(words_list, range(len(words_list))))

    words_onehot = [[] for _ in range(len(corpus))]

    for index, i in enumerate(all_words_list):
        for j in i:
            temp = [0 for _ in range(len(words_list))]
            temp[word_dict[j]] = 1
            words_onehot[index].append(temp)


    # print(words_list)
    # print(words_onehot)
    return words_list,words_onehot


def my_CountVectorizer(corpus):
    all_words_list,words_list = get_words(corpus)

    word_dict = dict(zip(words_list, [0] * len(words_list)))
    words_dict = [copy.deepcopy(word_dict) for _ in range(len(corpus))]

    for index, i in enumerate(all_words_list):
        for j in i:
            words_dict[index][j] += 1

    words_num_list = []

    for i in words_dict:
        words_num_list.append(list(i.values()))

    words_num_array = np.array(words_num_list)

    # print(words_list)
    # print(words_num_array)

    return words_list,words_num_array

def my_TfidfVectorizer(corpus):
    words_list,words_num_array = my_CountVectorizer(corpus)

    # DF
    words_num_iszero = words_num_array != 0
    det = np.sum(words_num_iszero, axis=0)

    # IDF
    idf = np.log((len(corpus)+1)/(det+1))+1

    # TF
    tf = words_num_array/np.sum(words_num_array,axis=1,keepdims=True)

    # TF-IDF
    tf_idf = tf*idf

    # print(tf_idf.round(2))
    return words_list,tf_idf




if __name__ == '__main__':
    # t1()
    # t0()
    t2()
    # corpus = [
    #     'This is the first document.',
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'Is this the first document?',
    # ]
    # my_ConvertSequence(corpus)
    # my_ConvertOnehot(corpus)
    # my_CountVectorizer(corpus)
    # my_TfidfVectorizer(corpus)

    pass
























