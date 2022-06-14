# Copyright (c) 2020 KU Leuven
import os
import sparsechem as sc
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import argparse
import os
import sys
import os.path
import time
import json
import functools
import csv
#from apex import amp
from contextlib import redirect_stdout
from sparsechem import Nothing
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from pytorch_memlab import MemReporter
import multiprocessing
from pynvml import *

def train():
    if torch.cuda.is_available():
        nvmlInit()

    multiprocessing.set_start_method('fork', force=True)


    parser = argparse.ArgumentParser(description="Training a multi-task model.")
    parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--y_class", "--y", "--y_classification", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--y_regr", "--y_regression", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--y_censor", help="Censor mask for regression (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks)", type=str, default=None)
    parser.add_argument("--weights_regr", "--weights_regression", help="CSV file with columns task_id, training_weight, censored_weight, aggregation_weight, aggregation_weight, task_type (for regression tasks)", type=str, default=None)
    parser.add_argument("--censored_loss", help="Whether censored loss is used for training (default 1)", type=int, default=1)
    parser.add_argument("--folding", help="Folding file (npy)", type=str, required=True)
    parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
    parser.add_argument("--fold_te", help="Test fold number (removed from dataset)", type=int, default=None)
    parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.02)
    parser.add_argument("--internal_batch_max", help="Maximum size of the internal batch", type=int, default=None)
    parser.add_argument("--normalize_loss", help="Normalization constant to divide the loss (default uses batch size)", type=float, default=None)
    parser.add_argument("--normalize_regression", help="Set this to 1 if the regression tasks should be normalized", type=int, default=0)
    parser.add_argument("--normalize_regr_va", help="Set this to 1 if the regression tasks in validation fold should be normalized together with training folds", type=int, default=0)
    parser.add_argument("--inverse_normalization", help="Set this to 1 if the regression tasks in validation fold should be inverse normalized at validation time", type=int, default=0)
    parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes of trunk", default=[], type=int, required=True)
    parser.add_argument("--last_hidden_sizes", nargs="+", help="Hidden sizes in the head (if specified , class and reg heads have this dimension)", default=None, type=int)
    #parser.add_argument("--middle_dropout", help="Dropout for layers before the last", type=float, default=0.0)
    #parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    parser.add_argument("--last_non_linearity", help="Last layer non-linearity (depecrated)", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--middle_non_linearity", "--non_linearity", help="Before last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="none", choices=["binarize", "none", "tanh", "log1p"])
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
    parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])
    parser.add_argument("--input_size_freq", help="Number of high importance features", type=int, default=None)
    parser.add_argument("--fold_inputs", help="Fold input to a fixed set (default no folding)", type=int, default=None)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--pi_zero", help="Reference class ratio to be used for calibrated aucpr", type=float, default=0.1)
    parser.add_argument("--min_samples_class", help="Minimum number samples in each class and in each fold for AUC calculation (only used if aggregation_weight is not provided in --weights_class)", type=int, default=5)
    parser.add_argument("--min_samples_auc", help="Obsolete: use 'min_samples_class'", type=int, default=None)
    parser.add_argument("--min_samples_regr", help="Minimum number of uncensored samples in each fold for regression metric calculation (only used if aggregation_weight is not provided in --weights_regr)", type=int, default=10)
    parser.add_argument("--dev", help="Device to use", type=str, default="cuda:0")
    parser.add_argument("--run_name", help="Run name for results", type=str, default=None)
    parser.add_argument("--output_dir", help="Output directory, including boards (default 'models')", type=str, default="models")
    parser.add_argument("--prefix", help="Prefix for run name (default 'run')", type=str, default='run')
    parser.add_argument("--verbose", help="Verbosity level: 2 = full; 1 = no progress; 0 = no output", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--save_model", help="Set this to 0 if the model should not be saved", type=int, default=1)
    parser.add_argument("--save_board", help="Set this to 0 if the TensorBoard should not be saved", type=int, default=1)
    parser.add_argument("--profile", help="Set this to 1 to output memory profile information", type=int, default=0)
    parser.add_argument("--mixed_precision", help="Set this to 1 to run in mixed precision mode (vs single precision)", type=int, default=0)
    parser.add_argument("--eval_train", help="Set this to 1 to calculate AUCs for train data", type=int, default=0)
    parser.add_argument("--enable_cat_fusion", help="Set this to 1 to enable catalogue fusion", type=int, default=0)
    parser.add_argument("--eval_frequency", help="The gap between AUC eval (in epochs), -1 means to do an eval at the end.", type=int, default=1)
    #hybrid model features
    parser.add_argument("--regression_weight", help="between 0 and 1 relative weight of regression loss vs classification loss", type=float, default=0.5)
    parser.add_argument("--scaling_regularizer", help="L2 regularizer of the scaling layer, if inf scaling layer is switched off", type=float, default=np.inf)
    parser.add_argument("--class_feature_size", help="Number of leftmost features used from the output of the trunk (default: use all)", type=int, default=-1)
    parser.add_argument("--regression_feature_size", help="Number of rightmost features used from the output of the trunk (default: use all)", type=int, default=-1)
    parser.add_argument("--last_hidden_sizes_reg", nargs="+", help="Hidden sizes in the regression head (overwritten by last_hidden_sizes)", default=None, type=int)
    parser.add_argument("--last_hidden_sizes_class", nargs="+", help="Hidden sizes in the classification head (overwritten by last_hidden_sizes)", default=None, type=int)
    parser.add_argument("--dropouts_reg", nargs="+", help="List of dropout values used in the regression head (needs one per last hidden in reg head, ignored if last_hidden_sizes_reg not specified)", default=[], type=float)
    parser.add_argument("--dropouts_class", nargs="+", help="List of dropout values used in the classification head (needs one per last hidden in class head, ignored if no last_hidden_sizes_class not specified)", default=[], type=float)
    parser.add_argument("--dropouts_trunk", nargs="+", help="List of dropout values used in the trunk", default=[], type=float, required=True)

    parser.add_argument("--optimizer", help="Choose the optimizer [Adam/SGD] (default: Adam)", type=str, default="Adam")
    parser.add_argument("--optimizer_params", nargs="+", help="Additional parameters for the optimizer in order as in Pytorch documentation eg. Adam: {beta1,  bete2, epsilon}, SGD:{momentum}. Default: Pytorch default.", default=[], type=float)


    args = parser.parse_args()


    if (args.last_hidden_sizes is not None) and ((args.last_hidden_sizes_class is not None) or (args.last_hidden_sizes_reg is not None)):
        raise ValueError("Head specific and general last_hidden_sizes argument were both specified!")
    if (args.last_hidden_sizes is not None):
        args.last_hidden_sizes_class = args.last_hidden_sizes
        args.last_hidden_sizes_reg   = args.last_hidden_sizes

    if args.last_hidden_sizes_reg is not None:
        assert len(args.last_hidden_sizes_reg) == len(args.dropouts_reg), "Number of hiddens and number of dropout values specified must be equal in the regression head!"
    if args.last_hidden_sizes_class is not None:
        assert len(args.last_hidden_sizes_class) == len(args.dropouts_class), "Number of hiddens and number of dropout values specified must be equal in the classification head!"
    if args.hidden_sizes is not None:
        assert len(args.hidden_sizes) == len(args.dropouts_trunk), "Number of hiddens and number of dropout values specified must be equal in the trunk!"

    def vprint(s=""):
        if args.verbose:
            print(s)

    vprint(args)


    args.optimizer = args.optimizer.lower()
    optim_suffix = ""
    if args.optimizer == "adam":
        if len(args.optimizer_params) > 0:
            if len(args.optimizer_params) == 3:
                vprint(f"Optimizer: Adam(beta1={args.optimizer_params[0]:.4f}, beta2={args.optimizer_params[1]:.4f}, epsilon={args.optimizer_params[2]:.4f})")
                optim_suffix=f"_Adam_1b{args.optimizer_params[0]}_2b{args.optimizer_params[1]}_eps{args.optimizer_params[2]}"
            else:
                raise ValueError("optimizer_params for Adam optimizer should have 3 values: beta1, beta2, epsilon")
        else:
            vprint(f"Optimizer: Adam(Default)")


    elif args.optimizer == "sgd":
        if len(args.optimizer_params) > 0:
            if len(args.optimizer_params) == 1:
                vprint(f"Optimizer: SGD(mementum = {args.optimizer_params[0]:.4f})")
                optim_suffix=f"_sgd_m{args.optimizer_params[0]}"
            else:
                raise ValueError("optimizer_params for SGD optimizer should have 1 value: momentum")
        else:
            vprint(f"Optimizer: SGD(Default)")
            optim_suffix="_sgd"
    else:
        raise ValueError("Unsupported optimizer! Supperted: Adam, SGD")
        

    if args.class_feature_size == -1:
        args.class_feature_size = args.hidden_sizes[-1]
    if args.regression_feature_size == -1:
        args.regression_feature_size = args.hidden_sizes[-1]

    assert args.regression_feature_size <= args.hidden_sizes[-1], "Regression feature size cannot be larger than the trunk output"
    assert args.class_feature_size <= args.hidden_sizes[-1], "Classification feature size cannot be larger than the trunk output"
    assert args.regression_feature_size + args.class_feature_size >= args.hidden_sizes[-1], "Unused features in the trunk! Set regression_feature_size + class_feature_size >= trunk output!"
    #if args.regression_feature_size != args.hidden_sizes[-1] or args.class_feature_size != args.hidden_sizes[-1]:
    #    raise ValueError("Hidden spliting not implemented yet!")


    if args.run_name is not None:
        name = args.run_name
    else:
        name  = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo_r{'.'.join([str(d) for d in args.dropouts_reg])}_wd{args.weight_decay}"
        name += f"_lr{args.lr}_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
        name += f"_fva{args.fold_va}_fte{args.fold_te}"
        name += optim_suffix
        if args.mixed_precision == 1:
            name += f"_mixed_precision"
    vprint(f"Run name is '{name}'.")

    if args.profile == 1:
        assert (args.save_board==1), "Tensorboard should be enabled to be able to profile memory usage."
    if args.save_board:
        tb_name = os.path.join(args.output_dir, "boards", name)
        writer  = SummaryWriter(tb_name)
    else:
        writer = Nothing()
    assert args.input_size_freq is None, "Using tail compression not yet supported."

    if (args.y_class is None) and (args.y_regr is None):
        raise ValueError("No label data specified, please add --y_class and/or --y_regr.")

    ecfp     = sc.load_sparse(args.x)
    y_class  = sc.load_sparse(args.y_class)
    y_regr   = sc.load_sparse(args.y_regr)
    y_censor = sc.load_sparse(args.y_censor)

    if (y_regr is None) and (y_censor is not None):
        raise ValueError("y_censor provided please also provide --y_regr.")
    if y_class is None:
        y_class = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
    if y_regr is None:
        y_regr  = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
    if y_censor is None:
        y_censor = scipy.sparse.csr_matrix(y_regr.shape)

    folding = np.load(args.folding)
    assert ecfp.shape[0] == folding.shape[0], "x and folding must have same number of rows"

    ## Loading task weights
    tasks_class = sc.load_task_weights(args.weights_class, y=y_class, label="y_class")
    tasks_regr  = sc.load_task_weights(args.weights_regr, y=y_regr, label="y_regr")

    ## Input transformation
    ecfp = sc.fold_transform_inputs(ecfp, folding_size=args.fold_inputs, transform=args.input_transform)
    print(f"count non zero:{ecfp[0].count_nonzero()}")
    num_pos    = np.array((y_class == +1).sum(0)).flatten()
    num_neg    = np.array((y_class == -1).sum(0)).flatten()
    num_class  = np.array((y_class != 0).sum(0)).flatten()
    if (num_class != num_pos + num_neg).any():
        raise ValueError("For classification all y values (--y_class/--y) must be 1 or -1.")

    num_regr   = np.bincount(y_regr.indices, minlength=y_regr.shape[1])

    assert args.min_samples_auc is None, "Parameter 'min_samples_auc' is obsolete. Use '--min_samples_class' that specifies how many samples a task needs per FOLD and per CLASS to be aggregated."

    if tasks_class.aggregation_weight is None:
        ## using min_samples rule
        fold_pos, fold_neg = sc.class_fold_counts(y_class, folding)
        n = args.min_samples_class
        tasks_class.aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)

    if tasks_regr.aggregation_weight is None:
        if y_censor.nnz == 0:
            y_regr2 = y_regr.copy()
            y_regr2.data[:] = 1
        else:
            ## only counting uncensored data
            y_regr2      = y_censor.copy()
            y_regr2.data = (y_regr2.data == 0).astype(np.int32)
        fold_regr, _ = sc.class_fold_counts(y_regr2, folding)
        del y_regr2
        tasks_regr.aggregation_weight = (fold_regr >= args.min_samples_regr).all(0).astype(np.float64)

    vprint(f"Input dimension: {ecfp.shape[1]}")
    vprint(f"#samples:        {ecfp.shape[0]}")
    vprint(f"#classification tasks:  {y_class.shape[1]}")
    vprint(f"#regression tasks:      {y_regr.shape[1]}")
    vprint(f"Using {(tasks_class.aggregation_weight > 0).sum()} classification tasks for calculating aggregated metrics (AUCROC, F1_max, etc).")
    vprint(f"Using {(tasks_regr.aggregation_weight > 0).sum()} regression tasks for calculating metrics (RMSE, Rsquared, correlation).")

    if args.fold_te is not None and args.fold_te >= 0:
        ## removing test data
        assert args.fold_te != args.fold_va, "fold_va and fold_te must not be equal."
        keep    = folding != args.fold_te
        ecfp    = ecfp[keep]
        y_class = y_class[keep]
        y_regr  = y_regr[keep]
        y_censor = y_censor[keep]
        folding = folding[keep]

    normalize_inv = None
    if args.normalize_regression == 1 and args.normalize_regr_va == 1:
       y_regr, mean_save, var_save = sc.normalize_regr(y_regr)
    fold_va = args.fold_va
    idx_tr  = np.where(folding != fold_va)[0]
    idx_va  = np.where(folding == fold_va)[0]

    y_class_tr = y_class[idx_tr]
    y_class_va = y_class[idx_va]
    y_regr_tr  = y_regr[idx_tr]
    y_regr_va  = y_regr[idx_va]
    y_censor_tr = y_censor[idx_tr]
    y_censor_va = y_censor[idx_va]

    if args.normalize_regression == 1 and args.normalize_regr_va == 0:
       y_regr_tr, mean_save, var_save = sc.normalize_regr(y_regr_tr) 
       if args.inverse_normalization == 1:
          normalize_inv = {}
          normalize_inv["mean"] = mean_save
          normalize_inv["var"]  = var_save
    num_pos_va  = np.array((y_class_va == +1).sum(0)).flatten()
    num_neg_va  = np.array((y_class_va == -1).sum(0)).flatten()
    num_regr_va = np.bincount(y_regr_va.indices, minlength=y_regr.shape[1])
    pos_rate = num_pos_va/(num_pos_va+num_neg_va)
    pos_rate_ref = args.pi_zero
    pos_rate = np.clip(pos_rate, 0, 0.99)
    cal_fact_aucpr = pos_rate*(1-pos_rate_ref)/(pos_rate_ref*(1-pos_rate))
    #import ipdb; ipdb.set_trace()
    batch_size  = int(np.ceil(args.batch_ratio * idx_tr.shape[0]))
    num_int_batches = 1

    if args.internal_batch_max is not None:
        if args.internal_batch_max < batch_size:
            num_int_batches = int(np.ceil(batch_size / args.internal_batch_max))
            batch_size      = int(np.ceil(batch_size / num_int_batches))
    vprint(f"#internal batch size:   {batch_size}")

    tasks_cat_id_list = None
    select_cat_ids = None
    if tasks_class.cat_id is not None:
        tasks_cat_id_list = [[x,i] for i,x in enumerate(tasks_class.cat_id) if str(x) != 'nan']
        tasks_cat_ids = [i for i,x in enumerate(tasks_class.cat_id) if str(x) != 'nan']
        select_cat_ids = np.array(tasks_cat_ids)
        cat_id_size = len(tasks_cat_id_list)
    else:
        cat_id_size = 0

    dataset_tr = sc.ClassRegrSparseDataset(x=ecfp[idx_tr], y_class=y_class_tr, y_regr=y_regr_tr, y_censor=y_censor_tr, y_cat_columns=select_cat_ids)
    dataset_va = sc.ClassRegrSparseDataset(x=ecfp[idx_va], y_class=y_class_va, y_regr=y_regr_va, y_censor=y_censor_va, y_cat_columns=select_cat_ids)

    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=dataset_tr.collate, shuffle=True)
    loader_va = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=dataset_va.collate, shuffle=False)

    args.input_size  = dataset_tr.input_size
    args.output_size = dataset_tr.output_size

    args.class_output_size = dataset_tr.class_output_size
    args.regr_output_size  = dataset_tr.regr_output_size
    args.cat_id_size = cat_id_size

    dev  = torch.device(args.dev)
    net  = sc.SparseFFN(args).to(dev)
    loss_class = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_regr  = sc.censored_mse_loss
    if not args.censored_loss:
        loss_regr = functools.partial(loss_regr, censored_enabled=False)

    tasks_class.training_weight = tasks_class.training_weight.to(dev)
    tasks_regr.training_weight  = tasks_regr.training_weight.to(dev)
    tasks_regr.censored_weight  = tasks_regr.censored_weight.to(dev)

    vprint("Network:")
    vprint(net)
    reporter = None
    h = None
    if args.profile == 1:
       torch_gpu_id = torch.cuda.current_device()
       if "CUDA_VISIBLE_DEVICES" in os.environ:
          ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
          nvml_gpu_id = ids[torch_gpu_id] # remap
       else:
          nvml_gpu_id = torch_gpu_id
       h = nvmlDeviceGetHandleByIndex(nvml_gpu_id)

    if args.profile == 1:
       #####   output saving   #####
       if not os.path.exists(args.output_dir):
           os.makedirs(args.output_dir)

       reporter = MemReporter(net)

       with open(f"{args.output_dir}/memprofile.txt", "w+") as profile_file:
            with redirect_stdout(profile_file):
                 profile_file.write(f"\nInitial model detailed report:\n\n")
                 reporter.report()

    if args.optimizer == "adam":
        if args.optimizer_params == []:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), 
                            betas = (args.optimizer_params[0], args.optimizer_params[1]), 
                            eps = args.optimizer_params[2], 
                            lr=args.lr, 
                            weight_decay=args.weight_decay)

    if args.optimizer == "sgd":
        if args.optimizer_params == []:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.optimizer_params[0], weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_alpha)

    num_prints = 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):
        t0 = time.time()
        sc.train_class_regr(
            net, optimizer,
            loader          = loader_tr,
            loss_class      = loss_class,
            loss_regr       = loss_regr,
            dev             = dev,
            weights_class   = tasks_class.training_weight * (1-args.regression_weight) * 2,
            weights_regr    = tasks_regr.training_weight * args.regression_weight * 2,
            censored_weight = tasks_regr.censored_weight,
            normalize_loss  = args.normalize_loss,
            num_int_batches = num_int_batches,
            progress        = args.verbose >= 2,
            reporter = reporter,
            writer = writer,
            epoch = epoch,
            args = args,
            scaler = scaler,
            nvml_handle = h)

        if args.profile == 1:
           with open(f"{args.output_dir}/memprofile.txt", "a+") as profile_file:
                profile_file.write(f"\nAfter epoch {epoch} model detailed report:\n\n")
                with redirect_stdout(profile_file):
                     reporter.report()

        t1 = time.time()
        eval_round = (args.eval_frequency > 0) and ((epoch + 1) % args.eval_frequency == 0)
        last_round = epoch == args.epochs - 1

        if eval_round or last_round:
            results_va = sc.evaluate_class_regr(net, loader_va, loss_class, loss_regr, tasks_class=tasks_class, tasks_regr=tasks_regr, dev=dev, progress = args.verbose >= 2, normalize_inv=normalize_inv, cal_fact_aucpr=cal_fact_aucpr)
       #     import ipdb; ipdb.set_trace()
            for key, val in results_va["classification_agg"].items():
                writer.add_scalar(key+"/va", val, epoch)
            for key, val in results_va["regression_agg"].items():
                writer.add_scalar(key+"/va", val, epoch)

            if args.eval_train:
                results_tr = sc.evaluate_class_regr(net, loader_tr, loss_class, loss_regr, tasks_class=tasks_class, tasks_regr=tasks_regr, dev=dev, progress = args.verbose >= 2)
                for key, val in results_tr["classification_agg"].items():
                    writer.add_scalar(key+"/tr", val, epoch)
                for key, val in results_tr["regression_agg"].items():
                    writer.add_scalar(key+"/tr", val, epoch)
            else:
                results_tr = None

            if args.verbose:
                ## printing a new header every 20 lines
                header = num_prints % 20 == 0
                num_prints += 1
                sc.print_metrics_cr(epoch, t1 - t0, results_tr, results_va, header)

        scheduler.step()

    #print("DEBUG data for hidden spliting")
    #print (f"Classification mask: Sum = {net.classmask.sum()}\t Uniques: {np.unique(net.classmask)}")
    #print (f"Regression mask:     Sum = {net.regmask.sum()}\t Uniques: {np.unique(net.regmask)}")
    #print (f"overlap: {(net.regmask * net.classmask).sum()}")
    writer.close()
    vprint()
    if args.profile == 1:
       multiplexer = sc.create_multiplexer(tb_name)
    #   sc.export_scalars(multiplexer, '.', "GPUmem", "testcsv.csv")
       data = sc.extract_scalars(multiplexer, '.', "GPUmem")
       vprint(f"Peak GPU memory used: {sc.return_max_val(data)}MB")
    vprint("Saving performance metrics (AUCs) and model.")

    #####   model saving   #####
    if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)

    model_file = f"{args.output_dir}/{name}.pt"
    out_file   = f"{args.output_dir}/{name}.json"

    if args.save_model:
       torch.save(net.state_dict(), model_file)
       vprint(f"Saved model weights into '{model_file}'.")

    results_va["classification"]["num_pos"] = num_pos_va
    results_va["classification"]["num_neg"] = num_neg_va
    results_va["regression"]["num_samples"] = num_regr_va

    if results_tr is not None:
        results_tr["classification"]["num_pos"] = num_pos - num_pos_va
        results_tr["classification"]["num_neg"] = num_neg - num_neg_va
        results_tr["regression"]["num_samples"] = num_regr - num_regr_va

    stats=None
    if args.normalize_regression == 1 :
       stats={}
       stats["mean"] = mean_save
       stats["var"]  = np.array(var_save)[0]
    sc.save_results(out_file, args, validation=results_va, training=results_tr, stats=stats)

    vprint(f"Saved config and results into '{out_file}'.\nYou can load the results by:\n  import sparsechem as sc\n  res = sc.load_results('{out_file}')")

if __name__ == "__main__":
    train()
