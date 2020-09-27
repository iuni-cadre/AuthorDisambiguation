# WoS-disambiguation
Project to compare and develop disambiguation solutions for Web fo Science and beyond


## Set up

First create a virtual environment for the project.

    conda create -n WoS-disambiguation python=3.7
    conda activate WoS-disambiguation

Install `ipykernel` for Jupyter and `snakemake` for workflow management. 

    conda install ipykernel
    conda install -c bioconda -c conda-forge snakemake

Create a kernel for the virtual environment that you can use in Jupyter lab/notebook.

    python -m ipykernel install --user --name WoS-disambiguation

Set up the config.yaml file, i.e., 
 
    cp workflow/config.template.yaml workflow/config.yaml

Then, set your password and username for the ElasticSearch server:

```
data_dir: "./data/"
es_password: "password for ElasticSearch"
es_username: "usename for ElasticSearch"
es_endpoint: "iuni2.carbonate.uits.iu.edu:9200/wos/_search/"
```

## Packages
- numpy
- scipy
- pandas
- dask
- 
