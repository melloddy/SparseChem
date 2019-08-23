import pandas as pd
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
import argparse

parser = argparse.ArgumentParser(description='Creating ChEMBL dataset.')
parser.add_argument('--max_num_atoms', help="Maximum number of atoms", type=int, default=80)
parser.add_argument("--salts", help="Salt removal", type=str, default="[Cl,Na,Br,I,K]")
args = parser.parse_args()

print(args)

#cl = np.load("./chembl_23_clusters_hier_0.6.npy")
smiles = pd.read_csv("./chembl_23_smiles_cleaned.csv")

## removing salt and creating the dataset
salt_remover = SaltRemover(defnData="[Cl,Na,Br,I,K]")

## only unique features for given ECFP radius
ecfp0 = []
ecfp2 = []
ecfp4 = []
ecfp6 = []

## converting smiles into ECFP
keep3 = np.zeros(smiles.shape[0], dtype=np.bool)

for i in tqdm.trange(smiles.shape[0]):
    mol = salt_remover.StripMol(Chem.MolFromSmiles(smiles.canonical_smiles.iloc[i]))
    if mol.GetNumAtoms() > args.max_num_atoms:
        continue
    keep3[i] = True
    fps0 = AllChem.GetMorganFingerprint(mol, 0).GetNonzeroElements().keys()
    fps1 = AllChem.GetMorganFingerprint(mol, 1).GetNonzeroElements().keys()
    fps2 = AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements().keys()
    fps3 = AllChem.GetMorganFingerprint(mol, 3).GetNonzeroElements().keys()
    ecfp0.append(np.array(list(fps0)))
    ecfp2.append(np.array(list( set(fps1) - set(fps0) )))
    ecfp4.append(np.array(list( set(fps2) - set(fps1) )))
    ecfp6.append(np.array(list( set(fps3) - set(fps2) )))

print(f"Kept {keep3.sum()} compounds out of {keep3.shape[0]}.")

def make_csr(ecfpx):
    ecfpx_lengths = [len(x) for x in ecfpx]
    ecfpx_cmpd    = np.repeat(np.arange(len(ecfpx)), ecfpx_lengths)
    ecfpx_feat    = np.concatenate(ecfpx)
    ecfpx_val     = np.ones(ecfpx_feat.shape)

    ecfpx_feat_uniq = np.unique(ecfpx_feat)
    fp2idx = dict(zip(ecfpx_feat_uniq, range(ecfpx_feat_uniq.shape[0])))
    ecfpx_idx     = np.vectorize(lambda i: fp2idx[i])(ecfpx_feat)

    X0 = csr_matrix((ecfpx_val, (ecfpx_cmpd, ecfpx_idx)))
    return X0

X0 = make_csr(ecfp0)
X2 = make_csr(ecfp2)
X4 = make_csr(ecfp4)
X6 = make_csr(ecfp6)

## how many higher than 10% frequency
X0mean = X0.mean(0)
X2mean = X2.mean(0)
X4mean = X4.mean(0)
X6mean = X6.mean(0)

print(f"Saved data for {len(md)} compounds into files with prefix '{args.prefix}'.")
