# SL-neural-network-for-temporal-graph

Stuart-Landau oscillatory neural network for temporal link prediction.

## Acknowledgements

This project builds upon and adapts code from the following repositories:

- TGAT - [https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)
- SL-GNN - [https://github.com/kevvzhang/StuartLandauGNN](https://github.com/kevvzhang/StuartLandauGNN)

I gratefully acknowledge the authors for making their code publicly available. Modifications have been made to integrate and extend these implementations for this project.

## Commands to run in Kaggle

`!git clone https://github.com/triptesh1212/SL-neural-network-for-temporal-graph.git`

`!mkdir log data saved_models saved_checkpoints`

`!wget http://snap.stanford.edu/jodie/wikipedia.csv -P data/`

`!python utils/process_data.py`

`!python -u models/tgat.py -d wikipedia --bs 200 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix hello_world`
