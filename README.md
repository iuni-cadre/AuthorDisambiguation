# WoS-disambiguation
Project to compare and develop disambiguation solutions for Web fo Science and beyond


## Set up

### Anaconda

First create a virtual environment for the project.

    conda create -n WoS-disambiguation python=3.7
    conda activate WoS-disambiguation

Install `ipykernel` for Jupyter and `snakemake` for workflow management. 

    conda install ipykernel
    conda install -c bioconda -c conda-forge snakemake

Create a kernel for the virtual environment that you can use in Jupyter lab/notebook.

    python -m ipykernel install --user --name WoS-disambiguation
