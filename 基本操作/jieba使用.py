import jieba
import os
import sys
import pandas as pd

print(sys.path)
print(jieba.__version__)
# jieba分词内部主要基于词典匹配实现，首先获取所有可能的分词方式，然后在所有分词方式选择最有可能的分词结果

def t1():
    # 自定义词典
    jieba.load_userdict("./jieba.word")
    # jieba.load_userdict(os.path.join(os.path.dirname(__file__), "jieba.word"))

    print(__file__)
    print(os.path.dirname(__file__))
    # 文本分词 动态规划
    print(" ".join(jieba.cut("我喜欢上学")))
    print(jieba.lcut("我喜欢上学"))
    print(jieba.lcut("送餐公司中饿了么是值得选择的"))

    a = jieba.cut("书话是嗲是不分i粉丝u啊辜负一个饿了吗")
    print(" ".join(a))

def t2():
    jieba.load_userdict("./text_classify.dict")

    datas = pd.read_csv("./datas/text_classify/train.csv", encoding="utf-8", header=None, sep="	")
    print(datas)

    datas2 = datas

    for i,data in enumerate(datas.iloc[:,0]):
        datas2.iloc[i,0] = " ".join(jieba.cut(data,HMM=True))
    print(datas2)

    datas2.to_csv("./datas/text_classify/train_tokens3.csv", encoding="utf-8",sep="	", index=False, header=False)

    print(("11".encode("utf-8")))

def t3():
    jieba.load_userdict("./text_classify.dict")

    with open("./datas/text_classify/train.csv","r",encoding="utf-8") as reader:
        with open("./datas/text_classify/train_tokens2.csv","w",encoding="utf-8") as writer:
            for line in reader:
                line,label = line.strip().split("\t")
                tokens = jieba.lcut(line)
                writer.write(" ".join(tokens)+"\t"+label+"\n")

if __name__ == '__main__':

    t1()

    pass
























