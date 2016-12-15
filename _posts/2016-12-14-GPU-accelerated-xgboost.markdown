---
layout: post
title:  "GPU Accelerated Xgboost"
date:   2016-12-14 00.00.00 -0800
author: Rory Mitchell
---

## GPU Accelerated Xgboost

Decision tree learning and gradient boosting have until recently been the domain of multicore CPUs. Here we showcase a new plugin providing GPU acceleration for the Xgboost library. 

The plugin can be found at:
https://github.com/dmlc/xgboost/tree/master/plugin/updater_gpu

# How fast is it?
The following benchmarks show a performance comparison against multicore CPUs for 500 boosting iterations. The new Pascal Titan X shows some nice performance improvements of up to 9.4x as compared to an i7 CPU. The Titan is also able to process the entire 10M row Higgs dataset in its 12GB of memory. 

Dataset | Instances | Features | i7-6700K | Titan X (pascal) | Speedup
--- | --- | --- | --- | --- | --- 
Yahoo LTR | 473,134 | 700 | 3738 | 507 | 7.37
Higgs | 10,500,000 | 28 | 31352 | 4173 | 7.51
Bosch | 1,183,747 | 968 | 9460 | 1009 | 9.38

We also tested the Titan X against a server with 2x Xeon E5-2695v2 CPUs (24 cores in total) on the Yahoo learning to rank dataset. The CPUs outperform the GPU by about 1.5x when using all available cores. This is still a nice result however considering the Titan X costs $1200 and the 2x Xeon CPUs cost almost $5000.

![](https://github.com/dmlc/web-data/raw/master/xgboost/gpu/yahooltr_xeon_titan.png)

# How does it work?
The xgboost algorithm requires scanning across gradient/hessian values and using these partial sums to evaluate the quality of splits at every possible split in the training set. The GPU xgboost algorithm makes use of fast parallel prefix sum operations to scan through all possible splits as well as parallel radix sorting to repartition data. It builds a decision tree for a given boosting iteration one level at a time, processing the entire dataset concurrently on the GPU.

The algorithm also switches between two modes. The first mode processes node groups in interleaved order using specialised multiscan/multireduce operations. This provides better performance at lower levels in the tree  when there are fewer leaf nodes. At later levels we switched to using radix sort to repartition the data and perform more conventional scan/reduce operations.

# How do I use it?
To use the GPU algorithm add the single parameter:
```python
# Python example
param['updater'] = 'grow_gpu'
```

Xgboost must be built from source using the cmake build system, following the instructions [here](https://github.com/dmlc/xgboost/tree/master/plugin/updater_gpu).

The plug-in may be used through the Python or CLI interfaces at this time. [A demo is available](https://github.com/dmlc/xgboost/tree/master/demo/gpu_acceleration) showing how to use the GPU algorithm to accelerate a cross validation task on a large dataset.

# About the author
The Xgboost GPU plugin is contributed by [Rory Mitchell](https://github.com/RAMitchell). The project was a part of a Masters degree dissertation at Waikato University.

