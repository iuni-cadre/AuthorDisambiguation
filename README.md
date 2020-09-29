# WoS-disambiguation
Project to compare and develop disambiguation solutions for Web fo Science and beyond


## Set up

### Environment
First create a virtual environment for the project.

    conda create -n WoS-disambiguation python=3.7
    conda activate WoS-disambiguation

Install `ipykernel` for Jupyter and `snakemake` for workflow management. 

    conda install ipykernel
    conda install -c bioconda -c conda-forge snakemake

Create a kernel for the virtual environment that you can use in Jupyter lab/notebook.

    python -m ipykernel install --user --name WoS-disambiguation

### Password and Username for ElasticSearch
Set up the config.yaml file, i.e., 
 
    cp workflow/config.template.yaml workflow/config.yaml

Then, set your password and username for the ElasticSearch server:

```
data_dir: "./data/"
es_password: "password for ElasticSearch"
es_username: "usename for ElasticSearch"
es_endpoint: "localhost:9200/wos/_search/"
```

Don't worry. The config.yaml is gitignored and won't be pushed to the remote. 

### Connect your computer to the ElasticSearch server

Establish the ssh tunneling:

```
username={your username in the server}
privatekey={your private key to ssh to the server}
server=iuni2.carbonate.uits.iu.edu
port=9200
ssh -i $privatekey -N -L $port:$username@localhost:$port $server
```

## Packages
- numpy
- scipy
- pandas
- dask

