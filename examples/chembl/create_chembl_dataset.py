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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 13})


parser = argparse.ArgumentParser(description='Creating ChEMBL dataset.')
parser.add_argument('--max_num_atoms', help="Maximum number of atoms", type=int, default=80)
parser.add_argument("--salt_removal", help="Salt removal", type=str, default="[Cl,Na,Br,I,K]")
args = parser.parse_args()

print(args)

#cl = np.load("./chembl_23_clusters_hier_0.6.npy")
smiles = pd.read_csv("./chembl_23_smiles_cleaned.csv")

## removing salt and creating the dataset
salt_remover = SaltRemover(defnData="[Cl,Na,Br,I,K]")

## only unique features for given ECFP radius
ecfp6 = []
ecfp6_counts = []

## converting smiles into ECFP
keep3 = np.zeros(smiles.shape[0], dtype=np.bool)

for i in tqdm.trange(smiles.shape[0]):
    mol = salt_remover.StripMol(Chem.MolFromSmiles(smiles.canonical_smiles.iloc[i]))
    if mol.GetNumAtoms() > args.max_num_atoms:
        continue
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
X6mean   = np.array(X6.mean(0)).flatten()

## filtering, and sorting based on distance from 0.5
top10pct = np.where((X6mean < 0.9) & (X6mean > 0.1))[0]
top10pct = top10pct[np.argsort(np.abs(0.5 - X6mean[top10pct]))]

## compute LSH
def make_lsh(X, bits):
    bit2int = np.power(2, np.arange(len(bits)))
    lsh     = X[:,bits] @ bit2int
    return lsh

## convert bool array 'bits' into string
# "".join(str(int(x)) for x in bits)

def fold_lsh(lsh, nfolds = 3):
    lsh_uniq = np.unique(lsh)
    lsh_fold = np.array_split(np.random.permutation(lsh_uniq), nfolds)
    lsh2fold = dict(zip(np.concatenate(lsh_fold), np.repeat(np.arange(nfolds), [len(f) for f in lsh_fold])))
    ## mapping lsh to folds
    return np.vectorize(lambda i: lsh2fold[i])(lsh)

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

## saving random split without clustering
fold_random = fold_lsh(np.arange(X6.shape[0]), nfolds = 3)
np.save("./chembl_23_folds_random.npy", fold_random)

## LSH clustered random split
secret = b"1i38vja09w29vja9a3rf98aj18c9q9afia94"

nbits = [14, 15, 16, 18, 20]
lshs  = [make_lsh(X6, top10pct[:i]) for i in nbits]
folds = [hashed_fold_lsh(lsh, secret, 3) for lsh in lshs]

df = pd.DataFrame({"ecfp": fps[top10pct], "chembl_frequency": X6mean[top10pct]})
df.to_csv("chembl_23_lsh_highest_entropy.csv")
print(f"Saved highest entropy features to 'chembl_23_lsh_highest_entropy.csv'.")

## saving covariance matrix
X6_top10 = X6[:, top10pct].todense()
corr     = np.corrcoef(X6_top10.T)
corr_df  = pd.DataFrame(corr)
corr_df.to_csv("chembl_23_lsh_corr.csv")

for i, n in enumerate(nbits):
    print(f"#clusters for {n}bits: {np.unique(lshs[i]).shape[0]}")

## saving folds
for i, n in enumerate(nbits):
    np.save(f"./chembl_23_folds{n}.npy", folds[i])

scipy.io.mmwrite("./chembl_23_ecfp6.mtx", X6)
print(f"Saved data for {X6.shape[0]} compounds into files with prefix 'chembl_23_ecfp6.mtx'.")

## plotting
def plot_size_hist(lsh, fname):
    unique, counts = np.unique(lsh, return_counts=True)
    fig, ax        = plt.subplots(1, 1)
    ax.hist(counts, np.arange(0, 1000.0, 10.))
    ax.set_xlabel("Cluster size")
    ax.set_ylabel("Count")
    ax.set_title(f"Total {len(unique)} clusters")
    ax.set_xlim((0, 1000))
    plt.savefig(fname)
    plt.close()

for i, n in enumerate(nbits):
    plot_size_hist(lshs[i], f"lsh_nbits{n}_sizes_hist.png")

## LF clusters
lf = np.load("/home/aarany/CmpdHackathon/chembl_23/chembl_23_clusters_hier_0.6.npy")
plot_size_hist(lf, f"lf_sizes_hist.png")

