# LifelongLearning

# 1. Brief introduction
1.1 Lifelong learning终生学习，又名continuous learning，increment learning，never ending learning。
通常机器学习中，单个模型只解决单个或少数几个任务，对于新的任务，我们一般重新训练新的模型。而LifeLong learning，则先在task1上使用一个模型，然后在task2上仍然使用这个模型，一直到task n。Lifelong learning探讨的问题是，一个模型能否在很多个task上表现都很好，如此下去，模型能力就会越来越强。
# 1.2 LifeLong learning需要解决的三个问题
1.2.1 Knowledge Retention 知识记忆。我们不希望学完task1的模型，在学习task2后，在task1上表现糟糕。也就是希望模型有一定的记忆能力，能够在学习新知识时，不要忘记老知识。但同时模型不能因为记忆老知识，而拒绝学习新知识。总之在新老task上都要表现比较好。
1.2.2 Knowledge Transfer 知识迁移。我们希望学完task1的模型，能够触类旁通，即使不学习task2的情况下，也能够在task2上表现不错。也就是模型要有一定的迁移能力。这个和transfer learning有些类似。
1.2.3 Model Expansion 模型扩张。一般来说，由于需要学习越来越多的任务，模型参数需要一定的扩张。但我们希望模型参数扩张是有效率的，而不是来一个任务就扩张很多参数。这会导致计算和存储问题。
