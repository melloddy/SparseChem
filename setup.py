# Copyright (c) 2020 KU Leuven
import setuptools

exec(open("sparsechem/version.py").read())

setuptools.setup(
    name="sparsechem",
    version=__version__,
    author="Jaak Simm",
    author_email="jaak.simm@gmail.com",
    description="SparseChem package",
    long_description="Fast and accurate machine learning models for biological and chemical models",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "scikit-learn", "tqdm", "torch==1.8.1", "tensorboard", "pytorch_memlab", "pynvml"],
)

