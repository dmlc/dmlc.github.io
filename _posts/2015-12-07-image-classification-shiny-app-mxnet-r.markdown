---
layout: post
title: "An Image Classification shiny App using MXNetR"
date:   2015-12-07 13:17:00 -0800
author: Qiang Kou
categories: rstats
comments: true
---

Early this week, Google announced its [Cloud Vision API](http://googlecloudplatform.blogspot.com/2015/12/Google-Cloud-Vision-API-changes-the-way-applications-understand-images.html), which can detect the content of an image.

It is still in the preview stage and very few people can use it.
But with the power of R and MXNet, you can try something very similar on your own laptop: an image classification shiny app.
Thanks to the powerful shiny framework, it is implemented with no more than 150 lines of R code.

![center](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_r_shiny/mxnetR.png)

### Installing mxnet package

Due to various reasons, `mxnet` package can't get on cran, but we try our best to make installation process easy.

For Windows and Mac users, you can install CPU-version of mxnet in R directly using the following code:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

If you want to use the power of your GPU or you are a Linux hacker, please follow [the link](http://mxnet.readthedocs.org/en/latest/build.html).

### Run the shiny app

Besides the `mxnet`, you will also need `shiny` for the web framework and `imager` for image preprocessing. Both of them are on CRAN, so you can install them easily.

```r
install.packages("shiny", repos="https://cran.rstudio.com")
install.packages("imager", repos="https://cran.rstudio.com")
```

The hardest part has been done if you finish all the installation.
Let's run the app and have fun!
You can clone [the repo](https://github.com/thirdwing/mxnet_shiny) or just use the line of code below in R.

```r
shiny::runGitHub("thirdwing/mxnet_shiny")
```

For the first time, it will take some time to download a pre-trained Inception-BatchNorm Network (you can know more details on network architecture from the [paper](http://arxiv.org/abs/1502.03167)). And then you can use your local figures or a url containing figure. Personally I think the result is quite good.

![center](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/blog_mxnet_r_shiny/mxnetR2.png)

### Code behind the app

You can find all the code from [the repo](https://github.com/thirdwing/mxnet_shiny).
Just like other shiny apps, we have `ui.R` and `server.R`. The `ui.R` is quite straightforward, we just define a `sidebarPanel` and a `mainPanel`.

Let's look into the `server.R`. All the web-related things are nicely handled by shiny. Besides that, there are 3 chunks of code.

First, we load the pre-trained model:

```r
model <<- mx.model.load("Inception/Inception_BN", iteration = 39)
synsets <<- readLines("Inception/synset.txt")
mean.img <<- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
```

Then we defined a function to preprocess figures:

```r
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  yy <- floor((shape[1] - short.edge) / 2) + 1
  yend <- yy + short.edge - 1
  xx <- floor((shape[2] - short.edge) / 2) + 1
  xend <- xx + short.edge - 1
  croped <- im[yy:yend, xx:xend,,]
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(croped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized)
  dim(arr) = c(224, 224, 3)
  # substract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}
```

Last, we read the figure and make prediction:

```r
im <- load.image(src)
normed <- preproc.image(im, mean.img)
prob <- predict(model, X = normed)
max.idx <- order(prob[,1], decreasing = TRUE)[1:5]
result <- synsets[max.idx]
```


If you met any problem, please just open [an issure](https://github.com/dmlc/mxnet/issues). And all PR will be truly appreciated!
