# Memory Profile

There is also an option to profile the memory usage on GPU while training sparsechem. 

# How to use Memory Profile option

To enable memory profiling the following option needs to be activated when running `train.py`:

```
  --profile PROFILE     Set this to 1 to output memory profile information
```
This will enable an extra output after the training step has finished:
```
Peak GPU memory used: 3630.65185546875MB
```
For more details an extra file `memprofile.txt` is also produced in the ouput folder. In this file you can have more details on the different parts of the memory usage, for example:
```
Storage on cuda:0
Tensor148                                               (1,)   512.00B
Tensor149                                               (1,)   512.00B
Tensor150                                       (6805, 2890)    37.51M
Tensor151                                         (2, 61205)   956.50K
Tensor152                                           (61205,)   239.50K
Tensor153                                           (61205,)   120.00K
Tensor154                                               (1,)   512.00B
Tensor155                                               (1,)   512.00B
Tensor1                                              (2890,)    11.50K
net.2.net.2.weight                              (2890, 6000)    66.15M
net.2.net.2.weight.grad                         (2890, 6000)    66.15M
net.2.net.2.bias                                     (2890,)    11.50K
net.2.net.2.bias.grad                                (2890,)    11.50K
Tensor82                                                (1,)   512.00B
Tensor83                                                (1,)   512.00B
Tensor67                                       (32000, 6000)   732.42M
Tensor69                                             (6000,)    23.50K
Tensor71                                        (2890, 6000)    66.15M
Tensor73                                             (2890,)    11.50K
Tensor66                                       (32000, 6000)   732.42M
Tensor68                                             (6000,)    23.50K
Tensor70                                        (2890, 6000)    66.15M
Tensor72                                             (2890,)    11.50K
net.0.net_freq.weight                          (32000, 6000)   732.42M
net.0.net_freq.weight.grad                     (32000, 6000)   732.42M
net.0.net_freq.bias                                  (6000,)    23.50K
net.0.net_freq.bias.grad                             (6000,)    23.50K
-------------------------------------------------------------------------------
Total Tensors: 857309726        Used Memory: 3.16G
The allocated memory on cuda:0: 3.47G
Memory differs due to the matrix alignment or invisible gradient buffer tensors
-------------------------------------------------------------------------------
``` 
You can also open tensorboard using:
```
tensorboard --logdir <outputdir>
```
and opening 
```
http://localhost:6006/#scalars&_smoothingWeight=0
```
