from setuptools import setup, find_packages

setup(
    name="dynamic-esm",
    version="1.0.0",
    description="Bridging Evolutionary Protein Language Models with Quantum-Informed Molecular Dynamics for Generalizable Binding Affinity Prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="yuanyangshi",

    url="https://github.com/yuanyangshi/dynamic-esm",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "pytorch-lightning>=2.0.0",
        "fair-esm>=2.0.0",
        "rdkit>=2022.9.5",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "einops>=0.6.0",
        "h5py>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
