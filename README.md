# README

Repository structure
```shell
icd10-lab/
│
├── data/                     # Raw & processed datasets (DVC controlled)
├── experiments/              # Experiment configuration YAMLs/JSONs
├── logs/                     # Logs & experiment outputs
├── models/                   # Trained models, serialized
├── notebooks/                # Exploration, EDA (not for production runs)
├── scripts/                  # CLI scripts for training, evaluation
├── src/
│   ├── benchmarks/           # Benchmark loader
│   ├── data_utils.py         # Dataset loading + processing functions
│   ├── experiment_runner.py  # Orchestrate single experiment
│   ├── metrics.py            # Performance metrics
│   └── solvers/              # Solver modules for classification
├── tests/                    # Integration and unit tests 
├── .dvc/                     # DVC config and cache
├── LEADERBOARD.md            # Leaderboard of best solvers
├── requirements.yml          # Conda environment file
└── README.md
```


## 1. Create Virtual Environment and Setup MLFlow

### Case 1: Local Run
Create, activate and install missing packages via `conda` and download spacy model for English language:
```shell
conda env create -f requirements.yml
python -m spacy download en_core_web_sm
```

Launch MLFlow server on port 8080 or any other free port:

```shell
mlflow server --host 127.0.0.1 --port 8080
```

### Case 2: HPC Cluster Run

The script `setup_conda_env.sh` will clone the gpu environment of the cluster and complement it with additional packages. The commands of the script are
```shell
conda create --name env-icd10-lab --clone gpulab
conda env update --file requirements.yml --prune
python -m spacy download en_core_web_sm
```
So, make it executable and run it with:
```shell
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

On frontend node run MLFlow server
```shell
mlflow ui --backend-store-uri file:~/mlruns --host 127.0.0.1 --port 5000
```

Set up SSH port forwarding on local machine (here local port 5000 forwarded to remote 5000)
```shell
ssh -L 5000:127.0.0.1:5000 maho12@s-sc-frontend3.charite.de
```

# 2. Start MLFlow User Interface


```shell
mlflow ui
```
and type into browser `http://localhost:5000` 

## NLP Preprocessing
Depending on the chunking strategy defined in the config you need additional packages for parsetree generation. The complete list of required packages can be installed via;
```shell
conda env update -f nlp-requirements.yml
```
Certain chunking strategies require the Stanford CoreNLP parser, which must be downloaded manually and placed unzipped into the git root directory under `packages`.

    1. Verify that Java is installed
    2. Create `packages` directory under `icd10-lab`
    3. Download latest CoreNLP parser found (here)[https://stanfordnlp.github.io/CoreNLP]

## DVC Data Control
```shell
pip install "dvc[gdrive]"
dvc remote add -d mygdrive gdrive://<folder-id>
dvc push
```

## Run Unit Tests
Install additional requirements:
```shell
conda env update -f dev-requirements.yml
```

E.g., to run data loader tests from `test_data.py`, run in terminal:
```shell
export PYTHONPATH=.
pytest tests/benchmarks/test_data.py
```