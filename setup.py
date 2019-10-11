import setuptools

setuptools.setup(
    name="sparsechem",
    version="0.2.0",
    author="Jaak Simm",
    author_email="jaak.simm@gmail.com",
    description="SparseChem package",
    long_description="Fast and accurate machine learning models for biological and chemical models",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "sklearn", "tqdm", "tensorboardX"],
    )

