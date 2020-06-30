import os
import shutil
import subprocess
import sparsechem as sc
import numpy as np
import string
import glob
from urllib.request import urlretrieve

def download_chembl23(data_dir="test_chembl23", remove_previous=False):
    if remove_previous and os.path.isdir(data_dir):
        os.rmdir(data_dir)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    files = ["chembl_23_x.npy", "chembl_23_y.npy", "folding_hier_0.6.npy"]
    url   = "https://www.esat.kuleuven.be/~jsimm/"
    for f in files:
        if not os.path.isfile(os.path.join(data_dir, f)):
            print(f"Downloading '{f}' into '{data_dir}'.")
            urlretrieve(f"{url}{f}", os.path.join(data_dir, f))

def random_str(size):
    return "".join([string.ascii_lowercase[i] for i in np.random.randint(0, 26, size=12)])

def test_classification(data_dir="test_chembl23", rm_output=True):
    download_chembl23()
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"

    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23_x.npy" +
        f" --y_class ./{data_dir}/chembl_23_y.npy" +
        f" --folding ./{data_dir}/folding_hier_0.6.npy" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 4" +
        f" --lr 1e-3" +
        f" --lr_steps 3" +
        f" --verbose 1"
    )

    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    conf_file  = glob.glob(f"{output_dir}/*.json")[0]
    model_file = glob.glob(f"{output_dir}/*.pt")[0]

    results = sc.load_results(conf_file)

    assert os.path.isdir(os.path.join(output_dir, "board"))
    assert "conf" in results
    assert "validation" in results

    assert results["validation"]["classification"].shape[0] > 0

    if rm_output:
        shutil.rmtree(output_dir)

def test_noboard(data_dir="test_chembl23", rm_output=True):
    download_chembl23()
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23_x.npy" +
        f" --y_class ./{data_dir}/chembl_23_y.npy" +
        f" --folding ./{data_dir}/folding_hier_0.6.npy" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 1" +
        f" --save_board 0" +
        f" --verbose 0"
    )
    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    assert os.path.isdir(os.path.join(output_dir, "board")) == False
    if rm_output:
        shutil.rmtree(output_dir)

def test_regression(data_dir="test_chembl23", rm_output=True):
    download_chembl23()
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"

    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23_x.npy" +
        f" --y_regr ./{data_dir}/chembl_23_y.npy" +
        f" --folding ./{data_dir}/folding_hier_0.6.npy" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 4" +
        f" --lr 1e-3" +
        f" --lr_steps 3" +
        f" --verbose 1"
    )

    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    conf_file  = glob.glob(f"{output_dir}/*.json")[0]
    model_file = glob.glob(f"{output_dir}/*.pt")[0]

    results = sc.load_results(conf_file)

    assert "conf" in results
    assert "validation" in results

    assert results["validation"]["regression"].shape[0] > 0

    if rm_output:
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_classification()
    test_regression()
