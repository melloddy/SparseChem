import sparsechem as sc
import numpy as np
from scipy import sparse
from ACE_ECE_calculation_forSparseChem_finalVersion import calculateErrors
import argparse


#Number of bins in args
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/SelectedTargetsWithMoreThanNMeasurements/BooleanArray_TargetsWithMoreThan4ActivesInactivesInEachFold.npy')
y_class = sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
#y_hat  = sparse.csr_matrix(np.load('/home/rosa/git/sc_bayesianLayer/sc_bayesianlayer/SparseChem_BayesianLayer.py', allow_pickle=True))
y_hat=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/sc_run_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')

#select correct fold for class dataset
folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
keep    = np.isin(folding, 0)
y_class = sc.keep_row_data(y_class, keep) 

ECE_list, ACE_list=calculateErrors(y_class[:, boolean], y_hat[:, boolean], 10)

print('ece: ', np.mean(ECE_list), np.std(ECE_list))
print('ace: ', np.mean(ACE_list), np.std(ECE_list))