---
layout: post
title:  "MinPy：剑气双修的武功秘籍"
date:   2017-01-18 00.00.00 -0800
author: 王敏捷
---

> MinPy基于MXNet引擎，提供NumPy的接口，以达到最灵活的编程能力和最高效的计算性能。

现在的深度学习系统就像五岳剑派，各大门派，互有所长，互有所短。不过从编程角度看，不外乎有「气宗」与「剑宗」之分。

## 深度学习的「剑」「气」之争

气宗讲究内家功夫，讲究「以气御剑」。外在形式并不重要，重要的是内在性能。在性能为王的如今，这也是很多门派所采纳的理念。远如五岳鼻祖之一的 Theano，近如目前的五岳盟主 Tensorflow，都采用符号式编程 (Symbolic Programming) 的模型。其核心思想是让用户通过编写符号来描述算法，算法描述完毕后再进行执行。由于深度学习算法往往是需要反复迭代的，系统可以静态地对算法进行优化，从而获得更好的执行性能。正所谓，「真气所至，草木皆是利剑」。只要系统提供的符号足够表达算法，用户就可以获得不错的性能。其问题也正在此。其一，符号式编程并不能涵盖所有的算法逻辑，特别是应对控制型逻辑（control dependency）显得笨拙。其二，由于需要掌握一门新的符号语言，对于新手的上手难度比较高。其三，由于有算法描述和执行两个阶段，如果算法逻辑和实际执行的值相关，符号式编程将比较难以处理。

相对来说，命令式编程（Imperative Programming）则更像剑宗。剑宗注重招式的灵活与变化。远如当年剑宗第一高手 NumPy，近如贵为五岳之一的 Torch 都是采用命令式编程的接口。他和符号式编程最大的不同在于，命令式编程并没有描述算法和执行两个阶段，因此用户可以在执行完一个语句后，直接使用该语句的结果。这对于深度学习算法的调试和可视化等都是非常重要的特性。命令式编程的缺点在于，由于算法是一边执行一边描述的，因此对算法的优化是一个挑战。

究竟是「以剑御气」还是「以气御剑」？其实两者应该相辅相成。如果你空有一身内力却无一丁点剑招，就会像是刚得到逍遥子毕生内力的虚竹，想到巧妙复杂的深度学习模型只能干瞪眼却无法实现。如果你空有华丽招式而不精进内力，别人以拙破巧，你优美的模型只会被别人用简单粗暴的高性能，大模型和大数据给击倒。正因如此，五岳新贵 MXNet 同时支持符号式和命令式编程接口。用户可以选择在性能优先的部分使用符号式编程，而在其余部分使用灵活性更高的命令式编程。不过这种分而治之的方式给用户带来了额外的选择负担，并没有将两者融汇贯通。因此，我们进一步基于 MXNet，开发了 MinPy，希望将这两者取长补短——使用命令式编程的接口，获得符号式编程的性能。

## MinPy 的剑宗招式

在编程接口上，MinPy 继承了剑宗第一高手 NumPy 老先生的精髓。正所谓「无招胜有招」。没有特殊语法的语法才是好语法。于是在使用 MinPy 时，你只需要简单改写一句 import 语句：

<center><pre style='color:#000000;background:#ffffff;'><span style='color:#800000; font-weight:bold; '>import</span> minpy<span style='color:#808030; '>.</span>numpy <span style='color:#800000; font-weight:bold; '>as</span> np
</pre></center>

就能够开始使用 MinPy 了。由于是完全的命令式编程的接口，编程的灵活性被大大提高。我们来看以下两个例子。


![Debug and print variables](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/p1.png)

<center><i>例 1: 调试和打印变量值</i></center>

在 Tensorflow 中，如果需要打印某个变量，需要在打印语句前加上 control_dependencies。因为如果没有这条语句，Print 这个运算并不会出现在所需要求的 x 变量的依赖路径上，因此不会被执行。而在 MinPy 中，我们保持了和 NumPy 一样的风格，因此可以直接使用 Python 原生的 print 语句。

![Data dependent branching](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/p2.png)

<center><i>例 2: 数据依赖的分支语句</i></center>

数据依赖的分支语句是符号编程的另一个难点。比如在 Tensorflow 中，每个 branch 是一个 lambda，而并非直接运算。其原因在于符号编程将算法描述和实际执行分为两个部分，在没有算出来 x 和 y 的值之前，是无法知道究竟会取哪个分支的。因此，用户需要将分支的描述写成 lambda，以便在能在运行时再展开。这些语法虽然细微，但是仍然会对初学者带来负担。相对的，在 MinPy 中，由于采用命令式编程的接口，所以一切原生的 if 语句都可以使用。除了以上这些编程方面的区别外，MinPy 还提供了以下功能。

### 招式一：动态自动求导

符号编程的一个巨大优势是能够自动求导。这原本是命令式编程的弱项，原因在上面的例子中也有所体现。由于命令式编程需要应对各类分支和循环结构，这让自动求导变得比较复杂。MinPy 采纳了一位西域奇人 Autograd 的招法来解决这一问题。方法也非常简单：首先，用户将需要求导的代码定义在一个函数中，这样通过分析函数参数和返回值我们就能知道自动求导的输入和输出；其次，MinPy 一边执行一边记录下执行的路径，在自动求导时只需要反向这一路径即可。通过这一方法，MinPy 可以支持对于各类分支和循环结构的自动求导：

![Code example 1](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/code_examples/code_example_1.png)

###  招式二：完整 NumPy 支持

MinPy 的目标是希望只修改 import 语句，就能将 NumPy 程序变成 MinPy 程序，从而能够使用 GPU 进行加速。无奈 NumPy 老先生的招式博大精深，接口繁多，MinPy 作为后辈不能在短时间内支持所有的接口。因此，MinPy 采用了一套折中的策略。当用户使用 np.func 的时候，MinPy 会检测所调用的 func 是否已经有 GPU 支持。如果有，则直接调用，否则会使用 NumPy 原有的实现。同时，MinPy 会负责一切 CPU 和 GPU 之间的内存拷贝，完全做到用户透明。

![Code example 2](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/code_examples/code_example_2.png)

### 招式三：与符号式编程的衔接

尽管命令式编程能灵活地应对各种复杂的算法逻辑，出于性能的考虑，我们仍然希望对某些运算（特别是卷积运算）能够使用已有的符号执行的方式去描述。在 MinPy 中，我们也同样支持 MXNet 的符号编程。其思想是让用户将符号「包装」成一个函数进行调用。

![Code example 3](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/code_examples/code_example_3.png)

在上面这个例子中，我们将一个 Convolution 的符号包装成了一个函数。之后该函数可以像普通函数一样被反复调用。其中有一点需要注意的是，由于符号编程需要在执行前确定所有输入矩阵的大小，因此在上面例子中的 x 的大小不能任意改变。

## MinPy 的气宗修为

如之前所说，光有招式没有内功修为是没有办法成为令狐冲的，最多也就是个成不忧。命令式编程的挑战就在于如何优化算法使得性能能和符号式编程程序相较。以下我们比较了 MinPy 和使用 MXNet 符号编程的性能区别。

![Benchmark](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/benchmark.png)

在上面的例子中，我们测试了训练50层MLP网络的性能。我们分别比较了MXNet符号编程，与MinPy命令式编程的运行时间。结果可以看到当网络计算量比较大时，MinPy的命令式编程和符号编程的性能几乎相同。当计算量比较小时，命令式编程有明显性能差距。但如果在MinPy中使用符号编程则性能又和MXNet几乎相同。类似的，我们测试了训练RNN网络的性能。我们比较了MXNet的符号编程以及MinPy的命令式编程的性能区别。我们可以看到，在计算量比较大的情况下，命令式编程和符号式编程的性能比较接近。在小网络中，MinPy有一个固定的性能开销。我们认为这一性能开销主要来源于用于求导的动态路径记录，以及过细的计算粒度等问题。这些都是命令式编程所带来的性能挑战，也是MinPy今后的努力方向。

## 面向武林新人的武功宝典

对于想要加入五岳剑派的新人们，MinPy 也是一个非常适合的上手工具。原因之一是因为 MinPy 和 NumPy 完全兼容，几乎没有额外修改的语法。另一个原因是我们团队还提供了完整的 MinPy 版本的 CS231n 课程代码。CS231n 是斯坦福大学著名教授 Fei-Fei Li 和她的爱徒 Andrej Karpathy、Justin Johnson 讲授的一门深度学习入门课程。该课程完整覆盖各类深度学习基本知识，包括卷积神经网络和递归神经网络。该课程的作业并不仅仅是对这些知识点的简单堆砌，更是包含了很多最新的实际应用。由于 MinPy 和 NumPy 天生的界面相似性，我们团队改进了 CS231n，使得学生能够更好地体验如何在实际中训练和使用深度神经网络，也让学生能够体会到 MinPy 在实际研究环境下的便利性。基于 MinPy 的 CS231n 课件已经在上海科技大学和交通大学深度学习教程中被试用。

## 总结

团队从早期的 Minerva 项目开始，加入 MXNet 团队，陆续贡献了执行引擎、IO、Caffe 兼容 Op 等核心代码。MinPy 是我们回归用户界面，对纯命令式编程下的一次尝试。我们希望将最灵活的接口呈现给用户，而将最复杂的系统优化交给我们。MinPy 拥有和 NumPy 完全一致的接口，支持任意分支与循环的自动求导，以及良好的性能。MinPy 将进一步优化其性能，并即将成为 MXNet 项目的一部分。

## 链接

* Github 地址：[https://github.com/dmlc/minpy]()
* MinPy 文档地址：[http://minpy.readthedocs.io/en/latest/]()

## 鸣谢

* MXNet 开发社区
* 上海科技大学马毅教授、博士后周旭；上海交通大学俞凯教授、张伟楠老师
* 上海纽约大学博士生 Sean Welleck，本科生盖宇，李牧非

## MinPy 剑客名单

![People](https://raw.githubusercontent.com/dmlc/web-data/master/minpy/people.png)

*：MinPy 工作在 NYU Shanghai intern 期间完成。




