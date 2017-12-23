---
layout: post
title: "RNN made easy with MXNet R"
date:   2017-10-11
author: Jeremie Desgagne-Bouchard
categories: rstats
comments: true
---

This tutorial presents an example of application of RNN to text classification using padded and bucketed data to efficiently handle sequences of varying lengths. Some functionalities require running on a CUDA enabled GPU. 

Example based on sentiment analysis on the [IMDB data](http://ai.stanford.edu/~amaas/data/sentiment/).

What's special about sequence modeling?
---------------------------------------

Whether working with times series or text at the character or word level, modeling sequences typically involves dealing with samples of varying length.

To efficiently feed the Recurrent Neural Network (RNN) with samples of even length within each batch, two tricks can be used:

-   Padding: fill the modeled sequences with an arbitrary word/character up to the longest sequence. This results in sequences of even lengths, but potentially of excessive size for an efficient training.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/pad-1.png)![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/pad-2.png)

-   Bucketing: apply the padding trick to subgroups of samples split according to their lengths. It results in multiple training sets, or buckets, within which all samples are padded to an even length. Diagram below illustrates how the two previous samples would be pre-processed if using buckets of size 4 and 6.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/bucket1-1.png)![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/bucket1-2.png)

Non numeric features such as words need to be transformed into a numeric representation. This task is commonly performed by the embedding operator which first requires to convert words into a 0 based index. The embedding will map a vector of features based on that index. In the example below, the embedding projects each word into 2 new numeric features.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/bucket2-1.png)![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/bucket2-2.png)

Data preparation
----------------

For this demo, the data preparation is performed by the script [data_preprocessing_seq_to_one.R](https://github.com/apache/incubator-mxnet/tree/master/example/rnn/bucket_R/data_preprocessing_seq_to_one.R) which involves the following steps:

-   Import IMDB movie reviews
-   Split each review into a word vector and apply some common cleansing (remove special characters, lower case, remove extra blank space...)
-   Convert words into integers and define a dictionary to map the resulting indices with former words
-   Aggregate the buckets of samples and labels into a list

To illustrate the benefit of bucketing, two datasets are created:

-   `corpus_single_train.rds`: no bucketing, all samples are padded/trimmed to 600 words.
-   `corpus_bucketed_train.rds`: samples split into 5 buckets of length 100, 150, 250, 400 and 600.

Below is the example of the assignation of the bucketed data and labels into `mx.io.bucket.iter` iterator. This iterator behaves essentially the same as the `mx.io.arrayiter` except that is pushes samples coming from the different buckets along with a bucketID to identify the appropriate symbolic graph to use.

``` r
corpus_bucketed_train <- readRDS(file = "data/corpus_bucketed_train.rds")
corpus_bucketed_test <- readRDS(file = "data/corpus_bucketed_test.rds")

vocab <- length(corpus_bucketed_test$dic)

### Create iterators
batch.size = 64

train.data.bucket <- mx.io.bucket.iter(buckets = corpus_bucketed_train$buckets, 
                                batch.size = batch.size, 
                                data.mask.element = 0, shuffle = TRUE)

eval.data.bucket <- mx.io.bucket.iter(buckets = corpus_bucketed_test$buckets, 
                                batch.size = batch.size, 
                                data.mask.element = 0, shuffle = FALSE)
```

Define the architecture
-----------------------

Below are the graph representations of a seq-to-one architecture with LSTM cells. Note that input data is of shape `seq.length X batch.size` while the RNN operator requires input of shape `hidden.features X batch.size X seq.length`, requiring to swap axis.

For bucketing, a list of symbols is defined, one for each bucket length. During training, at each batch the appropriate symbol is bound according to the bucketID provided by the iterator.

``` r
symbol_single <- rnn.graph(config = "seq-to-one", cell_type = "lstm", 
                           num_rnn_layer = 1, num_embed = 2, num_hidden = 4, 
                           num_decode = 2, input_size = vocab, dropout = 0.5, 
                           ignore_label = -1, loss_output = "softmax",
                           output_last_state = F, masking = T)
```

``` r
bucket_list <- unique(c(train.data.bucket$bucket.names, eval.data.bucket$bucket.names))

symbol_buckets <- sapply(bucket_list, function(seq) {
  rnn.graph(config = "seq-to-one", cell_type = "lstm", 
            num_rnn_layer = 1, num_embed = 2, num_hidden = 4, 
            num_decode = 2, input_size = vocab, dropout = 0.5, 
            ignore_label = -1, loss_output = "softmax",
            output_last_state = F, masking = T)})

graph.viz(symbol_single, type = "graph", direction = "LR", 
          graph.height.px = 50, graph.width.px = 800, shape=c(64, 5))
```

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/architect-1.png)

The representation of an unrolled RNN typically assumes a fixed length sequence. The operator `mx.symbol.RNN` simplifies the process by abstracting the recurrent cells into a single operator that accepts batches of varying length (each batch contains sequences of identical length).

Train the model
---------------

First the non bucketed model is trained for 6 epochs:

``` r
devices <- mx.gpu(0)

initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 2.5)

optimizer <- mx.opt.create("rmsprop", learning.rate = 1e-3, gamma1 = 0.95, gamma2 = 0.92, 
                           wd = 1e-4, clip_gradient = 5, rescale.grad=1/batch.size)

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)
batch.end.callback <- mx.callback.log.train.metric(period = 50)

system.time(
  model <- mx.model.buckets(symbol = symbol_single,
                            train.data = train.data.single, eval.data = eval.data.single,
                            num.round = 6, ctx = devices, verbose = FALSE,
                            metric = mx.metric.accuracy, optimizer = optimizer,  
                            initializer = initializer,
                            batch.end.callback = NULL, 
                            epoch.end.callback = epoch.end.callback)
)
```

    ##    user  system elapsed 
    ## 205.214  17.253 210.265

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/logger1-1.png)

Then training with the bucketing trick. Note that no additional effort is required: just need to provide a list of symbols rather than a single one and have an iterator pushing samples from the different buckets.

``` r
devices <- mx.gpu(0)

initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 2.5)

optimizer <- mx.opt.create("rmsprop", learning.rate = 1e-3, gamma1 = 0.95, gamma2 = 0.92, 
                           wd = 1e-4, clip_gradient = 5, rescale.grad=1/batch.size)

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)
batch.end.callback <- mx.callback.log.train.metric(period = 50)

system.time(
  model <- mx.model.buckets(symbol = symbol_buckets,
                            train.data = train.data.bucket, eval.data = eval.data.bucket,
                            num.round = 6, ctx = devices, verbose = FALSE,
                            metric = mx.metric.accuracy, optimizer = optimizer,  
                            initializer = initializer,
                            batch.end.callback = NULL, 
                            epoch.end.callback = epoch.end.callback)
)
```

    ##    user  system elapsed 
    ## 129.578  11.500 125.120

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/logger2-1.png)

The speedup is substantial, around 125 sec. instead of 210 sec., a 40% speedup with little effort.

Plot word embeddings
--------------------

Word representation can be visualized by looking at the assigned weights in any of the embedding dimensions. Here, we look simultaneously at the two embeddings learnt in the LSTM model.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_R_rnn_bucket/embed-1.png)

Since the model attempts to predict the sentiment, it's no surprise that the 2 dimensions into which each word is projected appear correlated with words' polarity. Positive words are associated with lower X1 values ("great", "excellent"), while the most negative words appear at the far right ("terrible", "worst"). By representing words of similar meaning with features of similar values, embedding much facilitates the remaining classification task for the network.  

Inference on test data
----------------------

The utility function `mx.infer.rnn` has been added to simplify inference on RNN with bucketed data.

``` r
ctx <- mx.gpu(0)
batch.size <- 64

corpus_bucketed_test <- readRDS(file = "data/corpus_bucketed_test.rds")

test.data <- mx.io.bucket.iter(buckets = corpus_bucketed_test$buckets, 
                              batch.size = batch.size, 
                              data.mask.element = 0, shuffle = FALSE)
```

``` r
infer <- mx.infer.rnn(infer.data = test.data, model = model, ctx = ctx)

pred_raw <- t(as.array(infer))
pred <- max.col(pred_raw, tie = "first") - 1
label <- unlist(lapply(corpus_bucketed_test$buckets, function(x) x$label))

acc <- sum(label == pred)/length(label)
roc <- roc(predictions = pred_raw[, 2], labels = factor(label))
auc <- auc(roc)
```

Accuracy: 87.6%

AUC: 0.9436
