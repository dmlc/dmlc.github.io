---
layout: post
title: "Use Caffe operator in MXNet"
date: 2016-07-30 00:00:00 +0900
author: Haoran Wang
categories: mxnet
comments: true
---

This blog demonstrates two steps to use Caffe operator in MXNet, including:

* How to install MXNet with Caffe support.

* How to embed Caffe's op into MXNet's symbolic graph.

## Install Caffe With MXNet interface
* Download offical Caffe repository [BVLC/Caffe](https://github.com/BVLC/caffe).
* Download [caffe's patch] (https://github.com/BVLC/caffe/pull/4527.patch) with mxnet-interface. Move patch file under your caffe folder and apply the patch by `git apply patch_file_name`.
* Install caffe following [official guide](http://caffe.berkeleyvision.org/installation.html).

## Compile with Caffe
* In mxnet folder, open `config.mk` (if you haven't already, copy `make/config.mk` (Linux) or `make/osx.mk` (Mac) into MXNet root folder as `config.mk`) and uncomment the lines `CAFFE_PATH = $(HOME)/caffe` and `MXNET_PLUGINS += plugin/caffe/caffe.mk`. Modify `CAFFE_PATH` to your caffe installation if necessary. 
* Run `make clean && make` to build with caffe support.

## Caffe Operator (Layer)
Caffe's neural network operator and loss func are supported by MXNet through `mxnet.symbol.CaffeOp` and `mxnet.symbol.CaffeLoss` respectively.
For example, the following code shows multi-layer perception network for classifying MNIST digits ([full code](https://github.com/dmlc/mxnet/blob/master/example/caffe/caffe_net.py)):
```Python
data = mx.symbol.Variable('data')
fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")
act1 = mx.symbol.CaffeOp(data_0=fc1, prototxt="layer{type:\"TanH\"}")
fc2  = mx.symbol.CaffeOp(data_0=act1, num_weight=2, name='fc2', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 64} }")
act2 = mx.symbol.CaffeOp(data_0=fc2, prototxt="layer{type:\"TanH\"}")
fc3 = mx.symbol.CaffeOp(data_0=act2, num_weight=2, name='fc3', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 10}}")
mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
```
Let's break it down. First `data = mx.symbol.Variable('data')` defines a variable as placeholder for input.
Then it's fed through Caffe's operators with `fc1  = mx.symbol.CaffeOp(data_0=data, num_weight=2, name='fc1', prototxt="layer{type:\"InnerProduct\" inner_product_param{num_output: 128} }")`.

The inputs to caffe op are named as data_i for i=0 ... num_data-1 as `num_data` is the number of inputs. You may skip the argument, as the example does, if its value is 1. While `num_weight` is number of `blobs_`(weights). Its default value is 0, as many ops maintain no weight. `prototxt` is the configuration string.

We could also replace the last line by:
```Python
label = mx.symbol.Variable('softmax_label')
mlp = mx.symbol.CaffeLoss(data=fc3, label=label, grad_scale=1, name='softmax', prototxt="layer{type:\"SoftmaxWithLoss\"}")
```
to use loss funciton in caffe.

## Use customized caffe operators
Running customized operator from mxnet is no difference than using regular ones. There's no need to add any code in mxnet, as mxnet directly calls caffe layer registry.

## Bio
Caffe-plugin is contributed by [Haoran Wang](https://github.com/HrWangChengdu). 

Haoran is an incoming master student of MCDS program at Carnegie Mellon University. He received his Bachelor degree in Computer Science from ACM Class at Shanghai Jiao Tong University.

He has many thanks to Minjie Wang, Tianqi Chen, Junyuan Xie and Prof. Zheng Zhang for their helpful advices on implementation and documentation of caffe-plugin.
