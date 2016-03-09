---
layout: post
title: "MXNet Scala Package Released"
date:   2016-03-09 00:36:00 +0800
author: Yizhi Liu
categories: mxnet
comments: true
---

I'm really glad to annouce the release of [MXNet Scala Package](https://github.com/dmlc/mxnet/tree/master/scala-package), which brings the very flexible and efficient deep learning framework to JVM.

With the Scala API, now you are able to integrate MXNet into your JVM stacks. Think about constructing state-of-art deep learning models in Scala, Java and other languages built on JVM, and applying them to tasks such as image classification and data science challenges. Besides, Scala/Java codes for tensor/matrix computation on multiple GPUs will be seemless and really easy to write.

This is not Zootopia, but try everything you like!

# Build

Checkout the [Installation Guide](http://mxnet.readthedocs.org/en/latest/build.html) contains instructions to install mxnet.
Then you can compile the Scala Package by

```bash
make scalapkg
```

Run unit/integration tests by

```bash
make scalatest
```

Easy, huh? If everything goes well, you will find a jar file named like `mxnet_2.10-osx-x86_64-0.1-SNAPSHOT-full.jar` under `assembly/target`. You can use this jar in your own project.

Currently we support Linux and OSX with Java 1.6+. We'll soon deploy the jars to [Maven Repository](http://mvnrepository.com/) and try to make it available on Windows.

# Usage

Here I give an example of how to train a 3-layer MLP on MNIST with Scala.

### Model Definition

The model definition is straightforward, almost the same as the one in Python/R/Juila package:

```scala
import ml.dmlc.mxnet._

// model definition
val data = Symbol.Variable("data")
val fc1 = Symbol.FullyConnected(name = "fc1")(Map("data" -> data, "num_hidden" -> 128))
val act1 = Symbol.Activation(name = "relu1")(Map("data" -> fc1, "act_type" -> "relu"))
val fc2 = Symbol.FullyConnected(name = "fc2")(Map("data" -> act1, "num_hidden" -> 64))
val act2 = Symbol.Activation(name = "relu2")(Map("data" -> fc2, "act_type" -> "relu"))
val fc3 = Symbol.FullyConnected(name = "fc3")(Map("data" -> act2, "num_hidden" -> 10))
val mlp = Symbol.SoftmaxOutput(name = "sm")(Map("data" -> fc3))
```

### Dataset

Now load the training data through IO module. Most of time, you need a piece of training data and a piece of validation data as well. Suppose that you have already downloaded and unpacked the [MNIST dataset](http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip):

```scala
val trainDataIter = IO.MNISTIter(Map(
  "image" -> "data/train-images-idx3-ubyte",
  "label" -> "data/train-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> batchSize.toString,
  "shuffle" -> "1",
  "flat" -> "0",
  "silent" -> "0",
  "seed" -> "10"))

val valDataIter = IO.MNISTIter(Map(
  "image" -> "data/t10k-images-idx3-ubyte",
  "label" -> "data/t10k-labels-idx1-ubyte",
  "data_shape" -> "(1, 28, 28)",
  "label_name" -> "sm_label",
  "batch_size" -> batchSize.toString,
  "shuffle" -> "1",
  "flat" -> "0", "silent" -> "0"))
```

### Training and Prediction

Choose proper training parameters and simply call `model.fit()`:

```scala
import ml.dmlc.mxnet.optimizer.SGD
// setup model
val model = new FeedForward(mlp, Context.cpu(), numEpoch = 10,
	optimizer = new SGD(learningRate = 0.1f, momentum = 0.9f, wd = 0.0001f))
model.fit(trainDataIter, valDataIter)
```

It shouldn't take too long for 'shallow' models like MLP. You can further do prediction using this model:

```scala
val probArrays = model.predict(valDataIter)
// in this case, we do not have multiple outputs
require(probArrays.length == 1)
val prob = probArrays(0)
// get predicted labels
val py = NDArray.argmaxChannel(prob)
// deal with predicted labels py
```

Here's a list of MNIST training benchmark we made on EC2 g2.8xlarge instance with scala-package:

|Model|Devices|KVStore|Epochs|Train Acc|Test Acc|Avg samples/sec|
|---|:---|:---:|---:|---:|---:|---:|
|MLP|1 cpu|local|10|0.9906|0.9740|7419.98|
|MLP|1 gpu|local|10|0.9925|0.9770|18572.50|
|Lenet|1 cpu|local|3|0.9916|0.9870|717.38|
|Lenet|1 gpu|local|3|0.9912|0.9867|1457.29|
|Lenet|4 cpus|local|10|0.9997|0.9920|1338.40|
|Lenet|4 gpus|local|10|0.9998|0.9909|7882.14|
|Lenet|4 gpus|local_allreduce|10|0.9998|0.9914|13696.00|

If you meet any problem, feel free to open [an issure](https://github.com/dmlc/mxnet/issues). Also we are looking for contributors to help us further improve the MXNet Scala Package. Any pull request will be highly appreciated!
