# Catalogue Fusion option

Pharma partners run commercial assays (e.g. safety panels, kinase selectivity assays) at CROs (e.g. Eurofins/CEREP, DiscoverX, …). 

<img src="docs/cat_pic1.png" width="300">

With Catalogue fusion partners have the option to fuse these tasks in a separate head:

<img src="docs/cat_pic2.png" width="400">

# How to use Catalogue Fusion

The tasks to be fused in a separate catalogue head should be specified in the weight file using an extra column `catalog_id`. For example:



```
task_id,task_type,training_weight,aggregation_weight,catalog_id
0.0,OTHER,1.0,0,
1.0,OTHER,1.0,0,
...
70.0,OTHER,1.0,1,ID0
...
86.0,OTHER,1.0,1,ID1
...
```

Now SparseChem knows which tasks will be fused while training. For prediction the same file could be provided as well so SparseChem knows when to use the Catalog prediction. 
For example:
```
python ../predict.py --x T11_x.npz --weights_class weights_clf_catid.csv --outprefix testing --model cls/sc_run_h800_ldo0.8_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.pt --conf cls/sc_run_h800_ldo0.8_wd1e-06_lr0.001_lrsteps10_ep2_fva2_fte0.json
```
If the weight file is not provided the Catalog predictions are ignored. The other steps remain the same as described [here](main.md).
