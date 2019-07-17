import pandas as pd
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
import argparse
import hashlib
import random

parser = argparse.ArgumentParser(description='Creating ChEMBL dataset.')
args = parser.parse_args()

print(args)

smiles = pd.read_csv("./chembl_23_smiles_cleaned.csv")
lsh_fps = pd.read_csv("./chembl_23_lsh_highest_entropy.csv")

lsh16  = lsh_fps.ecfp[0:16].values

## only unique features for given ECFP radius
ecfp6 = []
ecfp6_counts = []

## converting smiles into ECFP
keep3 = np.zeros(smiles.shape[0], dtype=np.bool)

for i in tqdm.trange(smiles.shape[0]):
    mol = Chem.MolFromSmiles(smiles.canonical_smiles.iloc[i])
    keep3[i] = True
    fps3 = AllChem.GetMorganFingerprint(mol, 3).GetNonzeroElements()
    ecfp6.append(np.array(list(fps3.keys())))
    ecfp6_counts.append(np.array(list(fps3.values())))

print(f"Kept {keep3.sum()} compounds out of {keep3.shape[0]}.")

def make_csr(ecfpx, ecfpx_counts):
    ecfpx_lengths = [len(x) for x in ecfpx]
    ecfpx_cmpd    = np.repeat(np.arange(len(ecfpx)), ecfpx_lengths)
    ecfpx_feat    = np.concatenate(ecfpx)
    ecfpx_val     = np.concatenate(ecfpx_counts)

    ecfpx_feat_uniq = np.unique(ecfpx_feat)
    fp2idx = dict(zip(ecfpx_feat_uniq, range(ecfpx_feat_uniq.shape[0])))
    ecfpx_idx       = np.vectorize(lambda i: fp2idx[i])(ecfpx_feat)

    X0 = csr_matrix((ecfpx_val, (ecfpx_cmpd, ecfpx_idx)))
    return X0, ecfpx_feat_uniq

X6, fps  = make_csr(ecfp6, ecfp6_counts)
X6.data  = X6.data.astype(np.int64)

def bits_to_str(bits):
    return "".join(str(int(x)) for x in bits)

lsh = [bits_to_str(np.isin(lsh16, x)) for x in tqdm.tqdm(ecfp6)]
lsh = np.array(lsh)

## convert bool array 'bits' into string
def sha256(inputs):
    m = hashlib.sha256()
    for i in inputs:
        m.update(i)
    return m.digest()

def lsh_to_fold(lsh, secret, nfolds):
    lsh_bin = str(lsh).encode("ASCII")
    h       = sha256([lsh_bin, secret])
    random.seed(h, version=2)
    return random.randint(0, nfolds - 1)

def hashed_fold_lsh(lsh, secret, nfolds = 3):
    ## map each lsh into fold
    lsh_uniq = np.unique(lsh)
    lsh_fold = np.vectorize(lambda x: lsh_to_fold(x, secret, nfolds=nfolds))(lsh_uniq)
    lsh2fold = dict(zip(lsh_uniq, lsh_fold))
    return np.vectorize(lambda i: lsh2fold[i])(lsh)

## LSH clustered random split
secret = b"1i38vja09w29vja9a3rf98aj14c9q9afia94"
num_folds = 5

folds = hashed_fold_lsh(lsh, secret, num_folds)
np.save("./chembl_23_folds.npy", folds)

## TODO: add activity data into Y (+1/-1 encoded)
np.save("./chembl_23_data.npy", {"X": X6, "Y": None})
print(f"Saved data for {X6.shape[0]} compounds into files with prefix 'chembl_23_ecfp6.npy'.")

