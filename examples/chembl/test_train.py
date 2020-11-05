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

    files = ["chembl_23mini_x.npy",
             "chembl_23mini_y.npy",
             "chembl_23mini_folds.npy",
             "chembl_23mini_class_weights.csv",
             "chembl_23mini_regr_weights.csv",
             "chembl_23mini_y_censored.npy"]
    url   = "https://www.esat.kuleuven.be/~jsimm/"
    for f in files:
        if not os.path.isfile(os.path.join(data_dir, f)):
            print(f"Downloading '{f}' into '{data_dir}'.")
            urlretrieve(f"{url}{f}", os.path.join(data_dir, f))

def create_weights(data_dir="test_chembl23"):
    df = pd.DataFrame({
        "task_id":         np.arange(100),
        "training_weight": np.clip(np.random.randn(100), 0, 1),
        "task_type":       np.random.choice(["adme", "panel", "other"], size=100),
    })
    df["aggregation_weight"] = np.sqrt(df.training_weight)
    df.to_csv(f"{data_dir}/chembl_23mini_class_weights.csv", index=False)

    ## censored weights for regression
    df["censored_weight"] = np.clip(np.random.randn(100), 0, 1)
    df.to_csv(f"{data_dir}/chembl_23mini_regr_weights.csv", index=False)

def random_str(size):
    return "".join([string.ascii_lowercase[i] for i in np.random.randint(0, 26, size=12)])

def test_classification(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"

    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_class ./{data_dir}/chembl_23mini_y.npy" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 2" +
        f" --lr 1e-3" +
        f" --lr_steps 1" +
        f" --dev {dev}" +
        f" --verbose 1"
    )

    download_chembl23(data_dir)
    res = subprocess.run(cmd.split())
    assert res.returncode == 0

    conf_file  = glob.glob(f"{output_dir}/*.json")[0]
    model_file = glob.glob(f"{output_dir}/*.pt")[0]

    results = sc.load_results(conf_file)

    assert os.path.isdir(os.path.join(output_dir, "boards"))
    assert "conf" in results
    assert "validation" in results

    assert results["validation"]["classification"].shape[0] > 0

    if rm_output:
        shutil.rmtree(output_dir)

def test_noboard(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_class ./{data_dir}/chembl_23mini_y.npy" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 1" +
        f" --save_board 0" +
        f" --dev {dev}" +
        f" --verbose 0"
    )
    download_chembl23(data_dir)
    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    assert os.path.isdir(os.path.join(output_dir, "boards")) == False
    if rm_output:
        shutil.rmtree(output_dir)

def test_regression(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_regr ./{data_dir}/chembl_23mini_y.npy" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 2" +
        f" --lr 1e-3" +
        f" --lr_steps 1" +
        f" --dev {dev}" +
        f" --verbose 1"
    )

    download_chembl23(data_dir)
    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    assert os.path.isdir(os.path.join(output_dir, "boards"))
    conf_file  = glob.glob(f"{output_dir}/*.json")[0]
    model_file = glob.glob(f"{output_dir}/*.pt")[0]

    results = sc.load_results(conf_file)

    assert "conf" in results
    assert "validation" in results

    assert results["validation"]["regression"].shape[0] > 0

    if rm_output:
        shutil.rmtree(output_dir)

def test_classification_regression(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_class ./{data_dir}/chembl_23mini_y.npy" +
        f" --y_regr ./{data_dir}/chembl_23mini_y.npy" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 2" +
        f" --lr 1e-3" +
        f" --lr_steps 1" +
        f" --dev {dev}" +
        f" --verbose 1"
    )

    download_chembl23(data_dir)
    res = subprocess.run(cmd.split())
    assert res.returncode == 0
    assert os.path.isdir(os.path.join(output_dir, "boards"))
    conf_file  = glob.glob(f"{output_dir}/*.json")[0]
    model_file = glob.glob(f"{output_dir}/*.pt")[0]

    results = sc.load_results(conf_file)

    assert "conf" in results
    assert "validation" in results

    assert results["validation"]["regression"].shape[0] > 0

    if rm_output:
        shutil.rmtree(output_dir)

def test_regression_censor(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_regr ./{data_dir}/chembl_23mini_y.npy" +
        f" --y_censor ./{data_dir}/chembl_23mini_y_censored.npy" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 2" +
        f" --lr 1e-3" +
        f" --lr_steps 3" +
        f" --dev {dev}" +
        f" --verbose 1"
    )

    download_chembl23(data_dir)
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

def test_regression_censor_weights(dev, data_dir="test_chembl23", rm_output=True):
    rstr = random_str(12)
    output_dir = f"./{data_dir}/models-{rstr}/"
    cmd = (
        f"python train.py --x ./{data_dir}/chembl_23mini_x.npy" +
        f" --y_regr ./{data_dir}/chembl_23mini_y.npy" +
        f" --y_censor ./{data_dir}/chembl_23mini_y_censored.npy" +
        f" --weights_regr ./{data_dir}/chembl_23mini_regr_weights.csv" +
        f" --folding ./{data_dir}/chembl_23mini_folds.npy" +
        f" --batch_ratio 0.1" +
        f" --output_dir {output_dir}" +
        f" --hidden_sizes 20" +
        f" --epochs 2" +
        f" --lr 1e-3" +
        f" --lr_steps 3" +
        f" --dev {dev}" +
        f" --verbose 1"
    )

    download_chembl23(data_dir)
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
    test_classification(dev="cuda:0")
    test_noboard(dev="cuda:0")
    test_regression(dev="cuda:0")
    test_regression_censor(dev="cuda:0")
    test_classification_regression(dev="cuda:0")
