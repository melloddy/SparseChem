import sparsechem as sc
import torch
from collections import OrderedDict
import argparse
import json

parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--model", help="Pytorch model file (.pt)", type=str, required=True)
parser.add_argument("--out_conf", help="Output config file", type=str, default="output_config.json")
parser.add_argument("--out_model", help="Output model file", type=str, default="output_model.pt")
args = parser.parse_args()

results_loaded = sc.load_results(args.conf, two_heads=True)

conf  = results_loaded["conf"]

#net = sc.SparseFFN(conf)

state_dict = torch.load(args.model)
#new attributes in v0.9.6, setting default
setattr(conf,"regression_feature_size",-1)
setattr(conf,"class_feature_size",-1)
if len(conf.hidden_sizes)==1:
   setattr(conf,"dropouts_trunk",[conf.last_dropout])
else:
   dropouts_trunk = []
   for i in range(len(conf.hidden_sizes)-1):
       dropouts_trunk.append(conf.middle_dropout)
   dropouts_trunk.apppend(conf.last_dropout)
   setattr(conf,"dropouts_trunk",dropouts_trunk)
delattr(conf,"middle_dropout")
last_dropout = conf.last_dropout
delattr(conf,"last_dropout")

if conf.last_hidden_sizes is not None:
    #only change config file dropouts and translate last_hidden_sizes
    dropouts_class=[]
    dropouts_regr=[]
    for i in range(len(conf.last_hidden_sizes)):
        dropouts_class.append(last_dropout)
        dropouts_regr.append(last_dropout)
    setattr(conf,"dropouts_class",dropouts_class)
    setattr(conf,"dropouts_reg",dropouts_regr)
    setattr(conf,"last_hidden_sizes_class", conf.last_hidden_sizes)
    setattr(conf,"last_hidden_sizes_reg", conf.last_hidden_sizes)

else:
    setattr(conf,"last_hidden_sizes_class", None)
    setattr(conf,"last_hidden_sizes_reg", None)
    setattr(conf,"dropouts_class", [])
    setattr(conf,"dropouts_reg", [])
net = sc.SparseFFN(conf)
state_dict_new = net.state_dict()
for key in state_dict.keys():
    if key not in ["classLast.net.2.weight","classLast.net.2.bias","regrLast.0.net.2.weight","regrLast.0.net.2.bias"]:
        state_dict_new[key] = state_dict[key]
    state_dict_new["classLast.net.initial_layer.2.weight"] = state_dict["classLast.net.2.weight"]
    state_dict_new["classLast.net.initial_layer.2.bias"] = state_dict["classLast.net.2.bias"]
    state_dict_new["regrLast.0.net.initial_layer.2.weight"] = state_dict["regrLast.0.net.2.weight"]
    state_dict_new["regrLast.0.net.initial_layer.2.bias"] = state_dict["regrLast.0.net.2.bias"]

net.load_state_dict(state_dict_new)

torch.save(net.state_dict(), args.out_model)
print(f"Saved model weights into '{args.out_model}'.")
out = dict()
out["conf"] = conf.__dict__
with open(args.out_conf, "w") as file_json:
    json.dump(out, file_json)
    print(f"Config file saved in {args.out_conf}")

