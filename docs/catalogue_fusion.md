# Catalogue Fusion option

Pharma partners run commercial assays (e.g. safety panels, kinase selectivity assays) at CROs (e.g. Eurofins/CEREP, DiscoverX, …). 

<img src="docs/cat_pic1.png" width="300">

With Catalogue fusion partners have the option to fuse these tasks in a separate head:

<img src="docs/cat_pic2.png" width="400">

# How to use Catalogue Fusion

The tasks to be fused in a separate catalogue head should be specified in the weight file using an extra column `cat_id`. For example:



```
task_id,task_type,training_weight,aggregation_weight,cat_id
0.0,OTHER,1.0,0,
1.0,OTHER,1.0,0,
...
70.0,OTHER,1.0,1,ID0
...
86.0,OTHER,1.0,1,ID1
...
```

Now SparseChem knows which tasks will be fused. The other steps remain the same as described [here](main.md).
