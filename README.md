# README

Repository structure
```shell
icd10-lab/
│
├── data/                     # Raw & processed datasets (DVC controlled)
├── experiments/              # Experiment configuration YAMLs/JSONs
├── models/                   # Trained models, serialized
├── notebooks/                # Exploration, EDA (not for production runs)
├── scripts/                  # CLI scripts for training, evaluation
├── src/
│   ├── data_utils.py         # Dataset loading + processing functions
│   ├── solvers/              # Solver modules for classification
│   ├── metrics.py            # Performance metrics
│   └── experiment_runner.py  # Orchestrate single experiment
├── logs/                     # Logs & experiment outputs
├── .dvc/                     # DVC config and cache
├── LEADERBOARD.md            # Leaderboard of best solvers
├── requirements.yml          # Conda environment file
└── README.md
```


## 1. Create Virtual Environment and Setup MLFlow

### Case 1: Local Run
Create, activate and install missing packages via `conda`:
```shell
conda env create -f requirements.yml
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