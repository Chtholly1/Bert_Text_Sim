## Bert家族的文本相似度
### 前言
文本相似度是一个比较宽泛的问题，因为不同业务场景下文本相似度的定义也不尽相同。   
另外，文本相似度从模型结构上又可分为两大类：
1. cross-encoder:把句子A,B进行拼接等操作后，一起塞到网络结构中进行encoder，最后输出二分类或者多分类结果。
2. Bi-encoder:对句子A,B分别进行encoder后，分别得到两个句子向量A_emb和B_emb，然后通过余弦相似度等方式计算二者的相似度。

这两种方法的各有优劣，cross的方式效果更好，bi的方式速度更快，而且可以离线储存doc的encoder向量，非常方便。
这个项目主要是记录一些主流的相似度模型（包括cross-encoder和bi-encoder），同时对比这些模型的效果。

### 文件层级介绍
* Bert_bi, Sentece_Bert模型
    * train_adv.py, 训练Sentence_BERT_regression
    * train_classification.py, 训练Sentence_BERT_classification
    * conf
        * config.py 修改超参数等
    * ..., 其他
* Bert_cross, Bert模型
    * train_adv.py, 训练Bert([CLS] A [SEP] B [SEP])
    * ..., 其他同上
* simCSE, simCSE模型
    * train_supervised.py, 训练simCSE有监督版模型
    * ..., 其他同上
* resource
    * base_models, 原始模型路径，本文所用[模型下载地址](https://huggingface.co/voidful/albert_chinese_base)
    * data, 训练数据路径，本文所用数据下载地址
### 正文
本文主要在Bert,Sentence_Bert和simCSE等三个比较主流的深度学习模型进行了文本相相似度实验。参考文献：   
1. [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
2. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)
3. [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)
#### 数据集
文本相似度的数据集就是一个难点，特别是实际业务中，大家一定要根据自身的业务和数据进行正样本和负样本的选择。
我们在这里用的是公开数据集SNLI（中文版）:   
`{'sentence1': '一个女人正走在街对面吃香蕉，而一个男人正紧跟在他的公文包后面。', 'sentence2': '一个女人吃了一根香蕉，穿过一条街，有一个男人跟在她后面。', 'gold_label': 'entailment'}
{'sentence1': '一个女人正走在街对面吃香蕉，而一个男人正紧跟在他的公文包后面。', 'sentence2': '一个吃香蕉的女人穿过街道', 'gold_label': 'entailment'}
{'sentence1': '一个女人正走在街对面吃香蕉，而一个男人正紧跟在他的公文包后面。', 'sentence2': '这个人的公文包是工作用的。', 'gold_label': 'neutral'}
{'sentence1': '一个人正领着一辆克莱德斯代尔(Clydesdale)走上一条干草路，在一个古老的乡村。', 'sentence2': '一个穿灰色西装的女人在喝茶。', 'gold_label': 'contradiction'}
{'sentence1': '两个女人，拿着带食物的容器，抱着。', 'sentence2': '两组敌对帮派成员互相吹毛求疵。', 'gold_label': 'contradiction'}
{'sentence1': '骑着马的人跳过一架坏了的飞机。', 'sentence2': '一个人在户外，骑着一匹马。', 'gold_label': 'entailment'}
{'sentence1': '高级时装女士们在城市里一群人旁边的电车外等着。', 'sentence2': '女人不在乎她们穿什么衣服。', 'gold_label': 'contradiction'}
{'sentence1': '一个人正领着一辆克莱德斯代尔(Clydesdale)走上一条干草路，在一个古老的', 'sentence2': '一个人正带着他的马走在乡间的路上。', 'gold_label': 'entailment'}
{'sentence1': '前景是白色的女人，后面是一个稍微落后的男人，背景是约翰的比萨和陀螺仪的标志。', 'sentence2': '那个女人在等一个朋友。', 'gold_label': 'neutral'}
{'sentence1': '几个人在餐馆里，其中一个是喝橙汁。', 'sentence2': '食客们在一家餐馆里。', 'gold_label': 'entailment'}`   
可以看到一共有entailment,neutral,contradiction三种标签，我们这里以entailment作为正样本，contradiction作为负样本。因此这个项目实际上解决的是一个二分类任务（相似or不相似）。   
至于评价指标，一般来说最常见的是F1值，但这个指标存在一个二分类任务常见的问题，就是我们需要去选择一个合适的阈值来作为正负样本的判断（并不是单纯的0.5），而不同的模型之间阈值肯定是不一致的，这就失去了评价的"一致性"。    
而在二分类任务中，AUC值反而是个很合适的指标，所以我**最终决定使用AUC**作为评价指标。

ps:
1. 最终的训练数据及测试数据中，正样本和负样本总是成对出现的。（即对于一个句子，若它有一个相似句作为正样本，则一定还有一个非相似句作为负样本）
#### 常规Albert
输入形如[cls] sent_A [sep] sent_B [sep] 格式，还是和之前的代码一样，这里不过多介绍Bert的原理。   
**模型效果**   

| 模型 | AUC |
| --- | --- |
| Albert(3w训练集合) | 0.9801 |
| Albert(6w训练集合) | 0.9840 |
|  |  |

#### Sentence-BERT
##### Regression Objective Function
分别求出sentece A、B的向量，然后进行cos乘积(-1,1)，最后以mse作为损失函数   
**模型效果**  
| 模型 | AUC |
| --- | --- |
| Albert(3w训练集合) | 0.8362 |

ps: 感觉这个方法不太符合预期，可能有些小bug

##### classification objective function
同样分别求出sentece A、B的向量，然后把EA,EB和|EA-EB|三个向量给拼接起来，然后过一个全连接层+softmax分类层，最后以CE作为损失函数    
**模型效果** 
| 模型 | AUC |
| --- | --- |
| Albert(3w训练集合) | 0.9486 |
| Albert(6w训练集合) | 0.9566 |
| Albert(10w训练集合) | 0.9625 |

#### simCSE(有监督)
**模型效果**    
| Col1 | Col2 |
| --- | --- |
| Albert(3w训练集合) | 0.8667 |
| Albert(6w训练集合) | 0.8802 |
| Albert(10w训练集合) | 0.887991 |


### 小结
本文在相同的数据集上，分别对Bert,Sentence-Bert和simCSE等三个比较主流的深度学习模型进行了一个初步的效果对比：   
从效果上来看：   
Bert> Sentece_Bert_classification > simCSE > Sentece_Bert_regression   
从效率上来看：   
simCSE == Sentece_Bert_regression > Sentece_Bert_classification > Bert   
可以看出来基本上效果和效率是成反比的。   
ps:
1. 从Sentece_Bert原论文来看，regression和classification在效果上是差不多的，但不知道为什么在个人实验上两者相差甚远。   
2. simCSE作者宣称其在有监督训练上明显好于Sentence_Bert，从实验结果来看SimCSE确实比和它效率一致的Sentece_Bert_regression的效果好，但比起Sentece_Bert_classification差了不少。
