# SL-neural-network-for-temporal-graph

Stuart-Landau oscillatory neural network for temporal link prediction.

## Acknowledgements

This project builds upon and adapts code from the following repositories:

- TGAT - [https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)
- SL-GNN - [https://github.com/kevvzhang/StuartLandauGNN](https://github.com/kevvzhang/StuartLandauGNN)

I gratefully acknowledge the authors for making their code publicly available. Modifications have been made to integrate and extend these implementations for this project.

## Commands to run in Kaggle

```{bash}
!git clone https://github.com/triptesh1212/SL-neural-network-for-temporal-graph.git
```

```{bash}
%cd SL-neural-network-for-temporal-graph
```

```{bash}
!mkdir log data saved_models saved_checkpoints
```

```{bash}
!wget http://snap.stanford.edu/jodie/wikipedia.csv -P data/
```

```{bash}
!python utils/process_data.py
```

```{bash}
!python -u models/tgat.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
```

```{bash}
!python -u models/sl_tgat.py -d wikipedia --bs 200 --uniform --n_degree 20 --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world
```



## How to run on the HPC

```{bash}
step ssh login <email> --provisioner cineca-hpc
```

```{bash}
ssh-keygen -R login.leonardo.cineca.it
```

```{bash}
ssh-keygen -F login.leonardo.cineca.it
```

```{bash}
ssh <user_id>@login.leonardo.cineca.it
```

```{bash}
cd $WORK
```

```{bash}
mkdir user_directory_x
```

```{bash}
cd user_directory_x
```

<br>

First load the Python 3.11 environment:

```{bash}
module load python/3.11.7
```

Set up a new Python virtual environment:

```{bash}
python3 -m venv $WORK/user_directory_x/venv
```

Activate your Python virtual environment before installing packages:

```{bash}
source $WORK/user_directory_x/venv/bin/activate
```

<br>

```{bash}
git clone https://github.com/triptesh1212/SL-neural-network-for-temporal-graph.git
```

```{bash}
cd SL-neural-network-for-temporal-graph
```

```{bash}
chmod +x run_sl_tgat.sh
```


```{bash}
sbatch run_sl_tgat.sh
```

```{bash}
squeue -u <user_id>
```

```{bash}
tail -f log/slurm_<>.out
```

```{bash}
cat log/slurm_<>.err
```

<br>

```{bash}
deactivate
```

```{bash}
exit
```







