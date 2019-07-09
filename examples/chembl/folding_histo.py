from scipy.sparse import csr_matrix, coo_matrix 
from matplotlib import pyplot as plt
import scipy.io as sio
def jaccard_dist(X1, X2):
    assert np.all(X1.data == 1), "Data should be binary"
    assert np.all(X2.data == 1), "Data should be binary"
    XXt = X1.dot(X2.T).todense()

    row_sums1 = X1.sum(axis = 1)
    row_sums2 = X2.sum(axis = 1)

    unions   = row_sums1 + row_sums2.T - XXt
    return 1.0 - XXt / unions


def jaccard_dist_sym(X):
    XXt = X.dot(X.T).todense()

    row_sums = XXt.diagonal()
    unions   = row_sums + row_sums.T - XXt
    return 1.0 - XXt / unions

def mindists(X, folds, Nsample= 1000, fold = 0):
    X0 =     X[folds == fold, :]
    Xother = X[folds != fold, :]
    refs =   X0[np.random.choice(X0.shape[0], Nsample, replace=False)]
    K = jaccard_dist(refs, Xother)
    return K.min(axis = 1)
#bins = np.linspace(0,0.9,37)
#ns,_ = np.histogram(near, bins)

X = sio.mmread(f'/home/jaak/git/sparsechem/examples/chembl/chembl_23_ecfp6.mtx')
X = X.tocsr()
for i in ["14", "15", "16", "18", "20"]:
    folds = np.load(f'/home/jaak/git/sparsechem/examples/chembl/chembl_23_folds{i}.npy')
    plt.hist(mindists(X, folds), np.arange(0, 1.01, 0.05))
    plt.xlabel("Distance")
    plt.savefig(f"folds{i}.pdf")
    plt.close()

X = sio.mmread(f'/home/aarany/CmpdHackathon/chembl_23/chembl_23_x.mtx')
X = X.tocsr()
folds = np.load('/home/aarany/CmpdHackathon/chembl_23/folding_hier_0.6.npy')
plt.hist(mindists(X, folds), np.arange(0, 1.01, 0.05))
plt.xlabel("Distance")
plt.savefig(f"folds_LF.pdf")
plt.close()
