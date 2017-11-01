# progressive-ai

Progressive AI

## Content

- [Whats the AI](#whats-the-ai)
- [What can be AI](#what-can-be-ai)
- [Machine Learning](#machine-learning)
- [Deep Learning](#deep-learning)
- [Real AI for ME](#real-ai-for-me)
- [AI Levels](#ai-levels)
- [Reinforcement Learning](#reinforcement-learning)
- [Meta Learning](#meta-learning)

## Whats the AI

What's the AI?

> -- From [Wiki](https://en.wikipedia.org/wiki/Artificial_intelligence): \
> Artificial intelligence (AI, also machine intelligence, MI) is apparently intelligent behaviour by machines, rather than the natural intelligence (NI) of humans and other animals. In computer science AI research is defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of success at some goal

> -- From [Baidu](https://baike.baidu.com/item/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/9180?fr=aladdin&fromid=25417&fromtitle=AI): \
> 人工智能（Artificial Intelligence），英文缩写为AI。它是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

## What can be AI

What can be AI? The machine learning? Deep learning? Or something others? Let's look at a picture shows the relationship between these 3 things:

![ai-ml-dl](https://github.com/zslucky/progressive-ai/blob/master/images/al-ml-dl.png)

Most of times we need an `agent` to do something like recognition or prediction. AI can do these thing well.

Now we used AI in many domains like `NLP(Natural Language Processing)自然语言处理`, `Compute Vision计算机视觉(image recognition, object detection, etc...)`, `Recommendation System推荐系统`, `Robot机器人` etc...

Each domain has its own algorithms, but some algorithms are the same.

**The Basic function?**

- data + probability and statistics - 数据 + 概率论和数理统计
- data/big data + machine learning - 数据/大数据 + 机器学习
- big data + deep learning - 大数据 + 深度学习

Above methods can be integrated with some big data framework and algorithms framework.

## Machine Learning

> -- From [Wiki](https://en.wikipedia.org/wiki/Machine_learning): \
> Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed

> -- From [Baidu](http://baike.baidu.com/link?url=vHyK-xsJBNq-kbu8c9ewuOuNQUdwaDGHhoEybClwhCI6dX_86cX975H-vdrT3-Iq6LqU5kcpuPKPHIttNCde0E94e-kPBzwj99JucJ3peWYwwuP_nVUhhL1LevwHzp87): \
> 机器学习(Machine Learning, ML)是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。

There are many algorithms like `SVM(support vector machine)`, `DT(decision tree)`, `bayes theory`, `KNN` etc... Most algorithms can be used together as ensemble, for example, we recognize a people whether is Bush? 1 for `true`, 0 for `false`, now we use 3 algorithms to recognize 5 pictures, see the results bellow:

```javascript
// Use SVM we got:
// 0 1 1 0 1
// Accuracy: 60%

// Use DT we got:
// 1 0 0 1 1
// Accuracy: 60%

//Use Bayes we got:
// 1 1 0 1 0
// Accuracy: 60%

// Use vote as assemble (random froest) we got:
// 1 1 0 1 1
// Accuracy: 80%
```

**Let's use `Eigenface` as an example:**

See the [demo](https://github.com/zslucky/ud120-projects/blob/master/pca/face%20detection.ipynb) which is provided by `sklearn` and modified a little by myself.

**For advanced:**

Actually, in order to detect face, we should solve many real-world issues, so many great teams provide many  different methods to solve complex issues, here is a general method which include 4 parts:

- Face detection(人脸定位)(e.g. Kalman filtering卡尔曼滤波等)
- Face alignment(人脸校准)(e.g. 3D/2D转换, 角度矫正等)
- Face verification(人脸确认)
- Face identifiction/recognition(人脸识别)

For more compute vision algorithms, please refer to `OpenCV` [official site](https://docs.opencv.org/master/d9/df8/tutorial_root.html)


## Deep Learning

> -- From [Wiki](https://en.wikipedia.org/wiki/Deep_learning): \
> Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on learning data representations, as opposed to task-specific algorithms. Learning can be supervised, partially supervised or unsupervised.

> -- From [Baidu](https://baike.baidu.com/item/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/3729729?fr=aladdin): \
> 深度学习的概念源于人工神经网络的研究。含多隐层的多层感知器就是一种深度学习结构。深度学习通过组合低层特征形成更加抽象的高层表示属性类别或特征，以发现数据的分布式特征表示。

It likes Machine learning, but use neural network. As data become bigger and bigger, hardware become more and more quick, deep learning become more and more popular.

Also there are many algorithms like `CNN(Convolutional Neural Network)卷积神经网络`, `RNN(Recurrent Neural Networks)循环神经网络`, `DRL`, `GAN`, etc...

Before these algorithms, we should know the `perceptron network(感知器神经网络)`, which we talked before, we can refer to these [article](http://neuralnetworksanddeeplearning.com/chap1.html) write by [Michael Nielsen](https://github.com/mnielsen)

## Real AI for ME

The introduction above shows that nowadays machine can do many things in our life.

**But do you think it's the real AI?**

The real AI for me is that it can do multi tasks, have thought.

**If it's not read AI, why introduce this?**

Because every things above has its own meaning at that time, all above are the AI's base.

## AI Levels

- Supervised learning(有监督学习)
- Unsupervised learning(无监督学习)
- Reinforcement learning(强化学习)
- Meta learning(元学习, 也有类似的概念如迁移学习等)

## Reinforcement Learning

An important theory is `Markov Decision Process(MDP)`. See the follow image:

![mdp](https://github.com/zslucky/progressive-ai/blob/master/images/reinforcement_learning_diagram.png)

MDP defined as a 5-tuple ***(S, A, P(·,·), R(·,·), γ)***

- ***S***: is a finite set of states. (一个有限的状态集)
- ***A***: is a finite set of actions (alternatively, ***A<sub>s</sub>*** is the finite set of actions available from state ***s*** ). ( ***A<sub>s</sub>*** 也可以代表是状态 ***s*** 中的可用的一个集合)
- ***P<sub>a</sub>(s, s’)*** : is the probability that action ***a*** in state ***s*** will lead to state ***s'*** . (在动作 ***a*** 导致状态 ***s*** 到 ***s'*** 的概率)
- ***R<span>a</span>(s, s’)*** : is the immediate reward (or expected immediate reward) received after transitioning from state ***s*** to state ***s'*** , due to action ***a***. (当 ***s*** 过渡到 ***s'*** 之后立即得到奖励值)
- ***γ ∈ [0,1]*** : is the discount factor, which represents the difference in importance between future rewards and present rewards. (衰减因子，表示当前奖励值与未来得到的奖励值的区别)

![mdp](https://github.com/zslucky/progressive-ai/blob/master/images/mdp.png)

**Here's the most popular algorithms divide into 4 types:**

![mdp](https://github.com/zslucky/progressive-ai/blob/master/images/rl-supports.png)

- **Model-free** : Not related to environment, action based on every reward. (不尝试去理解环境, 环境给什么就是什么，一步一步等待真实世界的反馈, 再根据反馈采取下一步行动)
- **Model-based** : Try to understand environment and build a model, try to use model to predict every reward, choose a best state then do action, it has a virtual environment.(尝试理解环境, 并创建一个模型, 利用模型来模拟未来的所有反馈, 选择最好的一个来进行行动, 他比Model-free多出了一个虚拟的环境, 有了想象力.)
- **Policy based** : According to environment, give all actions' probility, then used best 1. (根据环境直接给出所有行动的概率, 使用最佳的一个.)
- **Value based** : Give all action's value, next action is the biggest value action, it can't choose the continuous motion. (输出的是所有动作的价值, 根据最高价值来选动作，这类方法不能选取连续的动作.)
- **Monte-carlo update** : From game start to game end, consider all actions, then update it. (游戏开始后, 要等待游戏结束, 然后再总结这一回合中的所有转折点, 再更新行为准则.)
- **Temporal-difference update** : From game start, it will update at every step, so it can play the game as well as learning. (在游戏进行中每一步都在更新, 不用等待游戏的结束, 这样就能边玩边学习了.)
- **On-policy** : 必须本人在场, 并且一定是本人边玩边学习.
- **Off-policy** : 可以选择自己玩, 也可以选择看着别人玩, 通过看别人玩来学习别人的行为准则.

**Now let's use `Q-learning` as an example.**

see some base math expressions bellow:

 ***`π(s) = a`*** : policy function. (策略函数)

1 kind of the state value functions ***V<sub>π</sub>*** : (其中1种状态值函数, 实际中有很多种值函数, 这里将未来反馈奖励的总和作为值函数)

![mdp](https://github.com/zslucky/progressive-ai/blob/master/images/value-function.png)

## Meta Learning

Learning to Learn