# LifelongLearning

# 1. Brief Introduction
Lifelong learning终生学习，又名continuous learning，increment learning，never ending learning。

通常机器学习中，单个模型只解决单个或少数几个任务，对于新的任务，我们一般重新训练新的模型。而LifeLong learning，则先在task1上使用一个模型，然后在task2上仍然使用这个模型，一直到task n。Lifelong learning探讨的问题是，一个模型能否在很多个task上表现都很好，如此下去，模型能力就会越来越强。
# 2. LifeLong Learning 需要解决的三个问题
## 2.1 Knowledge Retention 知识记忆
我们不希望学完task1的模型，在学习task2后，在task1上表现糟糕。也就是希望模型有一定的记忆能力，能够在学习新知识时，不要忘记老知识。但同时模型不能因为记忆老知识，而拒绝学习新知识。总之在新老task上都要表现比较好。
## 2.2 Knowledge Transfer 知识迁移
我们希望学完task1的模型，能够触类旁通，即使不学习task2的情况下，也能够在task2上表现不错。也就是模型要有一定的迁移能力。这个和transfer learning有些类似。
## 2.3 Model Expansion 模型扩张
一般来说，由于需要学习越来越多的任务，模型参数需要一定的扩张。但我们希望模型参数扩张是有效率的，而不是来一个任务就扩张很多参数。这会导致计算和存储问题。
# 3. 发展历程
终身机器学习的概念大约是1995年由Thrun和Mitchell[1]提出的，主要有以下四个研究方向。
## 3.1 终身有监督学习
1. Thrun[2]率先研究了终身概念学习，即每个过去的或者新来的任务都是一个类或者概念。针对基于内存的学习和神经网络，出现了一些终身机器学习方法。
2. 文献[3]提出了利用终身学习提升神经网络的方法。
3. Fei等人[4]把终身学习扩展到累积学习（cumulative learning）。当遇到新的类别时，累积学习建立一个新的多类别分类器，它可以区分所有过去的和新的类别，也可以辨别测试集中的未知类别。这也为自学习（self-learning）奠定了基础，因为这种可以辨别未知类别的能力可以用来学习新的事物。
4. Ruvolo和Eaton [5]提出了高效的终身学习算法ELLA来提升多任务学习方法。
5. 陈等人[6]提出了一种针对朴素贝叶斯分类的终身学习技术。
6. Petina和Lampert等人[7]也对终身机器学习进行了理论研究。
## 3.2 终身无监督学习
1. 陈和刘等人[8]首次提出了终身主题模型。
2. 刘等人[9]提出了一种用于信息抽取的终身学习方法。Shu等人[12]针对情感挖掘问题提出了一种终身图标注方法来区分两类表情。
## 3.3 终身半监督学习
终身半监督学习的代表性工作是永动语言学习机系统（never-ending language learner，NELL）[10]
## 3.4 终身强化学习
Thrun和Mitchell[1]率先研究终身强化学习用于机器人学习。Tanaka和Yamamura[11]提出了一种终身强化学习方法，它把每个环境都看做一个任务。BouAmmar等人[12]提出了一种高效的策略梯度终身强化学习算法。
## 3.5 总结
尽管终身学习已有20多年的研究历史，但是目前为止还没有太多的研究。一个可能的原因是机器学习研究在过去20年主要关注统计和规则的方法。终身学习主要需要系统的方法。
然而，随着统计机器学习变得愈加成熟，研究者意识到它的局限性，终身学习将变得越来越重要。我们可以比较确信地说，如果没有终身学习的能力，即通过不断地积累已学到的知识并且用已有的知识以一种自激励的方式学习新的任务，我们不可能建立真正的智能系统，我们也仅能在一个很具体的领域解决问题。
# 4. 终身学习的方法的划分
## 4.1 正则化
其主要思想是「通过给新任务的损失函数施加约束的方法来保护旧知识不被新知识覆盖」，这类方法通常不需要用旧数据来让模型复习已学习的任务，因此是最优雅的一类增量学习方法。

Learning without Forgetting (ECCV 2016)[4]提出的LwF算法是基于深度学习的增量学习的里程碑之作。LwF算法是介于联合训练和微调训练之间的训练方式，LwF的特点是它不需要使用旧任务的数据也能够更新。LwF算法的主要思想来自于knowledge distillation，也就是使新模型在新任务上的预测和旧模型在新任务上的预测相近。

具体来说，LwF算法先得到旧模型在新任务上的预测值，在损失函数中引入新模型输出的蒸馏损失，然后用微调的方法在新任务上训练模型，从而避免新任务的训练过分调整旧模型的参数而导致新模型在旧任务上性能的下降。但是，这种方法的缺点是高度依赖于新旧任务之间的相关性，当任务差异太大时会出现任务混淆的现象(inter-task confusion)，并且一个任务的训练时间会随着学习任务的数量线性增长，同时引入的正则项常常不能有效地约束模型在新任务上的优化过程。

概括起来，基于正则化的增量学习方法通过引入额外损失的方式来修正梯度，保护模型学习到的旧知识，提供了一种缓解特定条件下的灾难性遗忘的方法。不过，虽然目前的深度学习模型都是过参数化的，但模型容量终究是有限的，我们通常还是需要在旧任务和新任务的性能表现上作出权衡。

其他各种不同的正则化手段：
Learning without Memorizing (CVPR 2019)
Learning a Unified Classifier Incrementally via Rebalancing (CVPR 2019)
Class-incremental Learning via Deep Model Consolidation (WACV 2020)
## 4.2 回放
在训练新任务时，一部分具有代表性的旧数据会被保留并用于模型复习曾经学到的旧知识，因此「要保留旧任务的哪部分数据，以及如何利用旧数据与新数据一起训练模型」，就是这类方法需要考虑的主要问题。

iCaRL: Incremental Classifier and Representation Learning (CVPR 2017)是最经典的基于回放的增量学习模型，iCaRL的思想实际上和LwF比较相似，它同样引入了蒸馏损失来更新模型参数，但又放松了完全不能使用旧数据的限制。

LwF在训练新数据时完全没用到旧数据，而iCaRL在训练新数据时为每个旧任务保留了一部分有代表性的旧数据(iCaRL假设越靠近类别特征均值的样本越有代表性)，因此iCaRL能够更好地记忆模型在旧任务上学习到的数据特征。

另外Experience Replay for Continual Learning (NIPS 2019)指出这类模型可以动态调整旧数据的保留数量，从而避免了LwF算法随着任务数量的增大，计算成本线性增长的缺点。

基于iCaRL算法的一些有影响力的改进算法包括End-to-End Incremental Learning (ECCV 2018)和Large Scale Incremental Learning (CVPR 2019)，这些模型的损失函数均借鉴了知识蒸馏技术，从不同的角度来缓解灾难性遗忘问题，不过灾难性遗忘的问题还远没有被满意地解决。

iCaRL的增量学习方法会更新旧任务的参数 ，因此很可能会导致模型对保留下来的旧数据产生过拟合，Gradient Episodic Memory for Continual Learning (NIPS 2017)针对该问题提出了梯度片段记忆算法(GEM)，GEM只更新新任务的参数而不干扰旧任务的参数，GEM以不等式约束的方式修正新任务的梯度更新方向，从而希望模型在不增大旧任务的损失的同时尽量最小化新任务的损失值。

GEM方向的后续改进还有Efficient Lifelong Learning with A-GEM (ICLR 2019)和Gradient based sample selection for online continual learning (NIPS 2019)。

另外，也有一些工作将VAE和GAN的思想引入了增量学习，比如Variational Continual Learning (ICLR 2018)指出了增量学习的贝叶斯性质，将在线变分推理和蒙特卡洛采样引入了增量学习，Continual Learning with Deep Generative Replay (NIPS 2017)通过训练GAN来生成旧数据，从而避免了基于回放的方法潜在的数据隐私问题，这本质上相当于用额外的参数间接存储旧数据，但是生成模型本身还没达到很高的水平，这类方法的效果也不尽人意。

总体来说，基于回放的增量学习的主要缺点是需要额外的计算资源和存储空间用于回忆旧知识，当任务种类不断增多时，要么训练成本会变高，要么代表样本的代表性会减弱，同时在实际生产环境中，这种方法还可能存在「数据隐私泄露」的问题。
## 4.3 参数隔离
参数隔离的想法比较粗暴，既然改变旧任务的参数会影响旧任务的性能，我们就不动旧任务的参数，增量扩大模型，将新旧任务的参数进行隔离。

PackNet (2018 CVPR) 是一种对新旧任务参数进行硬隔离的办法。每次新任务到来时，PackNet增量使用一部分模型空间，通过剪枝的方法保留冗余的模型空间，为下次任务留下余量。对于一个新任务，训练过程分两步，第一步，模型首先固定旧任务参数，使用整个模型训练当前任务，完成后利用剪枝去除一部分非重要参数。第二步，模型在剩下的参数空间中进行重新训练。显然，PackNet为每个任务分配一部分参数空间，限制了任务个数，对任务的顺序也带来了要求。

HAT(ICML 2018) 使用硬注意力(Hard Attention)机制根据不同任务对模型的不同部分进行遮盖(Mask)，从而为每个任务分配模型的不同部分。同时，HAT使用正则化项对注意力遮盖进行稀疏性约束，使得模型空间可以在任务间进行更好的分配。HAT方法期望使用注意力机制对模型空间进行了任务自适应的分配，可以更好地在任务间共享和隔离参数。
# 5. 重要的paper
## 5.1 Continual Lifelong Learning with Neural Networks: A Review (2019)
### 5.1.1 摘要

在这篇综述中，总结了与人工学习系统的continual/lifelong learning相关挑战，并比较了现有那些在不同程度上减轻catastrophic forgetting的NN方法。尽管NN在特定领域学习方面已取得了重大进展，但要在自动化智体和机器人上开发强大的lifelong learning，还需要进行大量研究。为此作者讨论了由生物系统中的lifelong learning因素所激发的各种研究，如structural plasticity、memory replay、curriculum & transfer learning、intrinsic motivation和multisensory integration等。

### 5.1.2
常规神经网络模型训练串行任务时候，新任务的学习会使先前学习的任务性能大大降低。尽管重新training from scratch从实用上解决了catastrophic forgetting，但效率极低，还阻碍了实时地学习新数据。

### 5.1.3
稳定性-可塑性（stability-plasticity）难题

### 5.1.4 减轻灾难性遗忘的尝试
1. 存储以前数据的存储系统，其定期重放那些与新数据抽取样本做交错的旧样本。
2. 设计专门的机制来保护合并的知识不被新信息的学习所覆盖。
3. 在连接主义（connectionist）模型，当要学习的新实例与先前观察的实例出现明显不同时，会发生灾难性遗忘。除了预定义足够量的神经资源，避免连接主义模型的灾难性遗忘有三个关键点：（i）为新知识分配额外的神经资源；（ii）如果资源固定，则使用非重叠的表示形式；（iii）把新信息和旧知识交织一起在表示中。

### 5.1.5 介绍并比较不同的神经网络方法来缓解不同程度的灾难性遗忘

从概念上讲，这些方法可以分为：
1）重新训练整个网络同时进行正则化防止以前学过的任务造成灾难性遗忘，
2）选择性训练网络并在必要时扩展以代表新网络，
3） 为记忆整合建模互补学习系统，例如用内存重放来合并内部表示。

1. 正则化方法（见4.1）
2. Dynamic Architectures方法对新信息的响应是通过动态适应新的神经资源改变体系结构属性，例如增加神经元或网络层进行重新训练。
3. CLS方法和内存重放（见4.2）

### 5.1.6 
在终身学习任务中这些算法的严格评估很少，为此作者讨论了使用和设计量化指标去测量大规模数据集的灾难性遗忘。 

### 5.1.7
希望从连续的sensorimotor体验中递增学习，以及多感官信息整合。

1. 发展性学习（Developmental learning）实时调节智体与环境的具体交互。
与提供大量信息的计算模型相反，发展性智体基于其sensorimotor经验以自主方式获得越来越复杂的技能。因此，分阶段发展（staged development），对于以较少的辅导经验来提升认知能力，至关重要。主动推理模型旨在了解如何通过动作和感知的双边使用来选择在动态和不确定环境下最能暴露原因的数据。

2. 有意义的方式组织示例，例如使学习任务难度逐渐变大，人和动物的学习性能会更好，即课程学习（curriculum learning）。
这激发了机器人技术中的类似方法和关于课程学习影响学习绩效的新机器学习方法。一些数据集（例如MNIST）进行的实验表明，课程学习可作为无监督的预训练，提高泛化能力，并加快训练收敛速度。但是，课程学习的有效性对于任务进展方式非常敏感。课程策略可以看作是迁移学习的特例，在初始任务收集的知识用来指导复杂的学习过程。

3. 迁移学习
前向迁移是指学习任务TA对未来任务TB性能的影响，而后向迁移是指当前任务TB对先前任务TA的影响。因此，假设同时学习多个学习任务以提高一项特定任务的性能，迁移学习代表了一种人工系统的重要功能，即从有限量的特定样本推断一般规律。迁移学习一直是机器学习和自主智体的一个开放的挑战。尽管说，通过编码个体、目标或场景元素不变关系信息的概念表示，抽象知识的迁移得以实现，但人们对大脑的特定神经机制调节高级迁移学习的了解却很少。 零样本学习和单样本学习在新任务上表现出色，但不能防止灾难性遗忘。

4. Intrinsic Motivation的计算模型从人类婴幼儿选择目标并逐步掌握技能的方式中获得启发，以此定义终身学习框架的发展结构。
内在动机的计算模型可以通过学习课程的在线（自）生成来收集数据并逐步获得技能。这允许通过主动控制复杂度的增长来有效和随机地选择学习任务。增强学习（reinforcement learning）的最新工作包括curiosity和Intrinsic Motivation，以解决奖励稀少或具有欺骗性的情况。在外部奖励非常稀疏的情况下，curiosity-driven exploration会提供内在的奖励信号，使智体能够自主地、逐步地学习日益复杂的任务。

5. 多感官处理（Multisensory Learning）是交叉模式激励的物理特性、先验知识、期望（例如，学习的相关性）、协同（scaffolding）感知、认知和行为之间相互作用的结果。
Multisensory Learning的过程在整个生命是动态的，会受到短期和长期变化的影响。它由外在和内在（exogenous & endogenous）因素的动态加权组成，这些因素决定了多模态如何相互影响。
## 5.2 “A continual learning survey: Defying forgetting in classification tasks“，2020年5月，作者来自KU Leuven和华为公司
### 5.2.1 摘要
这个综述专注于task incremental classification
本文主要工作包括：
1）最新技术的分类和总结，
2）使一个continual learner连续地确定stability-plasticity 折中的新框架，
3）对11种最先进的continual learning方法和4个baseline进行全面的实验比较。
## 5.3 “Class-incremental learning: survey and performance evaluation“，2020年
### 5.3.1 摘要
大多数早期的递增学习方法都考虑了任务递增学习（task-IL），算法可以在推理时访问task-ID。这样的方法不必区分来自不同任务的类。最近已经开始考虑类递增学习（class-IL），即学习者在推理时无法访问task-ID，因此必须能够区分所有任务的所有类。

该文对现有的递增学习方法进行了全面调查，尤其是对12种类递增方法进行了广泛的实验评估。作者考虑了几种实验方案，包括对多个大型数据集的类递增方法进行比较、对小型和大型domain shifts进行调查以及比较各种网络体系的结构等。

### 5.3.2类递增学习主要挑战的三类解决方案
1）基于正则化的解决方案，旨在最大程度地减少学习新任务对先前任务重要权重的影响；
2）基于示例的解决方案，存储有限的示例集以防止对先前任务的遗忘；
3）直接解决task-recency bias的解决方案，在类递增学习方法中，这是指对最近学习的任务偏差。
### 5.3.3 类递增学习发生灾难性遗忘的原因
1. 权重漂移（Weight drift）：在学习新任务时，将更新与旧任务相关的网络权重，以最大程度地减少新任务的损失。但是先前任务的性能会受到损害，通常会非常严重。
2. 激励漂移（Activation drift）：与权重漂移密切相关，改变权重会导致激活变化，进而导致网络输出变化。
3. 任务间混淆（Inter-task confusion）：在类递增学习，目标是将所有类与所有任务区分开。但是，由于永远不会共同训练类，网络权重无法最优区分所有类。
4. 任务出现偏差（Task-recency bias）：单独学习的任务可能具有无可比拟的分类器输出。通常最主要的任务偏差是针对较新的任务类，confusion matrices可以清楚地观察到这种效果。
### 5.3.4 解决前面灾难性遗忘的方法
1. regularization based methods
2. rehearsal-based methods
Incremental Classifier and Representation Learning (iCaRL)、内存、采样、任务平衡、排练和正则化组合。
3. bias-correction methods
Bias Correction (BiC)、End-to-End Incremental Learning (EEIL)，Learning a Unified Classifier Incrementally via Rebalancing (LUCIR)，Class-IL with dual memory (IL2M).
### 5.3.5 目前类递增学习新兴的趋势：
1. Exemplar learning 对样本进行参数设置并优化以防止遗忘。
2. Feature rehearsal 执行特征重放而不是图像重放，其中训练一个生成器在网络某个隐藏层生成特征。 这样，不想pseudo rehearsal那样，排练方法也可以应用于复杂的数据集。
3. Explicit task classification 在每个任务上学习一个classifier head，其只能区分任务内的类，而另一个分类器则预测task label。尚不清楚，为什么以及什么时候，显式任务分类优于一个联合分类器的学习。
4. Self- and unsupervised incremental learning 有一种方法，可以执行明确的任务分类并采用高斯混合模型拟合学习的表示。
5. Meta-learning 在解决相关任务时积累的信息用来学习新任务。 这种方法可以学习那些能减少未来梯度干扰并基于此改善传递的参数。

# Reference
1. Thrun S, Mitchell T M. Lifelong robot learning. In: Steels L,ed. The Biology and Technology of    Intelligent Autonomous Agents. Berlin: Springer,1995, 165–196
2. Thrun S. Is learning the n-th thing any easier than learning the first? Advances in Neural Information Processing Systems,1996: 640–646
3. Silver D L, Mercer R E. The task rehearsal method of life-long learning:overcoming impoverished data. In: Proceedings of the 15th Conference of the Canadian Society for Computational Studies of Intelligence on Advances in Artificial Intelligence. 2002, 90–101
4. Fei G L, Wang S, Liu B. Learning cumulatively to become more knowledgeable. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016, 1565–1574
5. Ruvolo P, Eaton E. ELLA: an efficient lifelong learning algorithm. International Conference on Machine Learning. 2013, 28(1): 507–515
6. Chen Z Y, Ma N Z, Liu B. Lifelong learning for sentiment classification. In: Proceedings of ACL Conference. 2015
7. Pentina A, Lampert C H. A PAC-Bayesian bound for lifelong learning. International Conference on Machine Learning. 2014: 991–999
8. Chen Z Y, Liu B. Topic modeling using topics from many domains, lifelong learning and big data. International Conference on Machine Learning, 2014
9. Liu Q, Liu B, Zhang Y L, Kim D S, Gao Z Q. Improving opinion aspect extraction using semantic similarity and aspect associations. In: Proceedings of the 30th AAAI Conference on Artificial Intelligence. 2016
10. Mitchell T, Cohen W, Hruschka E, Talukdar P, Betteridge J, Carlson A, Dalvi B, Gardner M, Kisiel B, Krishnamurthy J, Lao N, Mazaitis K, Mohamed T, Nakashole N, Platanios E, Ritter A, Samadi M, Settles B, Wang R, Wijaya D, Gupta A, Chen X, Saparov A, Greaves M, Welling J. Never-ending learning. In: Proceedings of the 29th AAAI Conference on Artificial Intelligence.2015, 2302–2310
11. Tanaka F, Yamamura M. An approach to lifelong reinforcement learning through multiple environments. In: Proceedings of the 6th European Workshop on Learning Robots. 1997, 93–9
12. BouAmmar H, Eaton E, Ruvolo P, Taylor M. Online multi-task learning for policy gradient methods. In: Proceedings of the 31st International Conference on Machine Learning. 2014, 1206–1214
