# LifelongLearning

# 1. Brief introduction
Lifelong learning终生学习，又名continuous learning，increment learning，never ending learning。
通常机器学习中，单个模型只解决单个或少数几个任务，对于新的任务，我们一般重新训练新的模型。而LifeLong learning，则先在task1上使用一个模型，然后在task2上仍然使用这个模型，一直到task n。Lifelong learning探讨的问题是，一个模型能否在很多个task上表现都很好，如此下去，模型能力就会越来越强。
# 2 LifeLong learning需要解决的三个问题
## 2.1 Knowledge Retention 知识记忆
我们不希望学完task1的模型，在学习task2后，在task1上表现糟糕。也就是希望模型有一定的记忆能力，能够在学习新知识时，不要忘记老知识。但同时模型不能因为记忆老知识，而拒绝学习新知识。总之在新老task上都要表现比较好。
## 2.2 Knowledge Transfer 知识迁移
我们希望学完task1的模型，能够触类旁通，即使不学习task2的情况下，也能够在task2上表现不错。也就是模型要有一定的迁移能力。这个和transfer learning有些类似。
## 2.3 Model Expansion 模型扩张
一般来说，由于需要学习越来越多的任务，模型参数需要一定的扩张。但我们希望模型参数扩张是有效率的，而不是来一个任务就扩张很多参数。这会导致计算和存储问题。
# 发展历程
终身机器学习的概念大约是1995年由Thrun和Mitchell[1]提出的，主要有以下四个研究方向。
## 终身有监督学习
1. Thrun[2]率先研究了终身概念学习，即每个过去的或者新来的任务都是一个类或者概念。针对基于内存的学习和神经网络，出现了一些终身机器学习方法。
2. 文献[3]提出了利用终身学习提升神经网络的方法。
3. Fei等人[4]把终身学习扩展到累积学习（cumulative learning）。当遇到新的类别时，累积学习建立一个新的多类别分类器，它可以区分所有过去的和新的类别，也可以辨别测试集中的未知类别。这也为自学习（self-learning）奠定了基础，因为这种可以辨别未知类别的能力可以用来学习新的事物。
4. Ruvolo和Eaton [5]提出了高效的终身学习算法ELLA来提升多任务学习方法。
5. 陈等人[6]提出了一种针对朴素贝叶斯分类的终身学习技术。
6. Petina和Lampert等人[7]也对终身机器学习进行了理论研究。
## 终身无监督学习
1. 陈和刘等人[8]首次提出了终身主题模型。
2. 刘等人[9]提出了一种用于信息抽取的终身学习方法。Shu等人[12]针对情感挖掘问题提出了一种终身图标注方法来区分两类表情。
## 终身半监督学习
## 终身强化学习
1. Thrun S, Mitchell T M. Lifelong robot learning. In: Steels L,ed. The Biology and Technology of    Intelligent Autonomous Agents. Berlin: Springer,1995, 165–196
2. Thrun S. Is learning the n-th thing any easier than learning the first? Advances in Neural Information Processing Systems,1996: 640–646
3. Silver D L, Mercer R E. The task rehearsal method of life-long learning:overcoming impoverished data. In: Proceedings of the 15th Conference of the Canadian Society for Computational Studies of Intelligence on Advances in Artificial Intelligence. 2002, 90–101
4. Fei G L, Wang S, Liu B. Learning cumulatively to become more knowledgeable. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016, 1565–1574
5. Ruvolo P, Eaton E. ELLA: an efficient lifelong learning algorithm. International Conference on Machine Learning. 2013, 28(1): 507–515
6. Chen Z Y, Ma N Z, Liu B. Lifelong learning for sentiment classification. In: Proceedings of ACL Conference. 2015
7. Pentina A, Lampert C H. A PAC-Bayesian bound for lifelong learning. International Conference on Machine Learning. 2014: 991–999
8. Chen Z Y, Liu B. Topic modeling using topics from many domains, lifelong learning and big data. International Conference on Machine Learning, 2014
9. Liu Q, Liu B, Zhang Y L, Kim D S, Gao Z Q. Improving opinion aspect extraction using semantic similarity and aspect associations. In: Proceedings of the 30th AAAI Conference on Artificial Intelligence. 2016
