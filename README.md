# Awesome Graph Data Augmentation and Attack Methods via Edge Perturbation
This repo contains comprehensive [statistics](#statistics) on graph data augmentation and attack methods for graph neural networks (GNNs) implemented via edge perturbation, together with the official [implementation](#implementation) of a plug-to-play module, termed edge priority detector (EPD), proposed by the paper <b>Revisiting Edge Perturbation for Graph Neural Network in Graph Data Augmentation and Attack </b>.



## Statistics
Edge perturbation is regarded as the fundamental modification to the graph structure. Recently, edge perturbation methods have been used in GNN-related domains to make augmentation or inject attacks to graphs. Typically, edge perturbation methods for GNNs propose to add or remove edges to enable topology-level modifications. The perturbed graph is then fed to a GNN for learning, which will cause distinct variations in model accuracy on downstream tasks. Existing literature on edge perturbation can be divided into two veins according to their purposes, that is, graph data augmentation and attack (Gaug and Gatk). We remark that graph-related tasks, including node classification, graph classification, and link prediction, are abbreviated as NC, GC, and LP, respectively.


<div align="center">
  
| Category | Method | Venue | Approach | Applied GNN Model | Task | Code |
| ---------- | ----------- | ----------- | ----------- | ----------- |  ----------- | ----------- |
| Gaug | [DropEdge](https://arxiv.org/abs/1907.10903) | ICLR'20 | Remove Edges | GCN, GraphSAGE, Deepgcns, ASGCN | NC | [Available](https://github.com/DropEdge/DropEdge) |
| Gaug | [NeuralSparse](https://proceedings.mlr.press/v119/zheng20d.html) | ICML'20 | Remove Edges | GCN, GraphSAGE, GAT, GIN | NC | - | 
| Gaug | [SGCN](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_22) | PAKDD'20 | Remove Edges | GCN, GraphSAGE | NC | [Available](https://github.com/shuaishiliu/SGCN) |
| Gaug | [AdaptiveGCN](https://dl.acm.org/doi/abs/10.1145/3459637.3482049) | CIKM'21 | Remove Edges | GCN, GraphSAGE, GIN | NC | [Available](https://github.com/GuangmingZhu/AdaptiveGCN) |
| Gaug | [PTDNet](https://dl.acm.org/doi/abs/10.1145/3437963.3441734) | WSDM'21 | Remove Edges | GCN, GraphSAGE, GAT | NC & LP | [Available](https://github.com/flyingdoog/PTDNet) |
| Gaug | [TADropEdge](https://arxiv.org/abs/2106.02892) | arXiv'21 | Remove Edges | GCN, GIN | NC & GC | -
| Gaug | [UGS](http://proceedings.mlr.press/v139/chen21p.html?ref=https://githubhelp.com) | ICML'21 | Remove Edges | GCN, GAT, GIN | NC & LP | [Available](https://github.com/VITA-Group/Unified-LTH-GNN)
| Gaug | [GAUG](https://ojs.aaai.org/index.php/AAAI/article/view/17315) | AAAI'21 | Add/Remove Edges | GCN, GAT, GraphSAGE, JK-NET | NC | [Available](https://github.com/zhao-tong/GAug)
| Gaug | [AdaEdge](https://ojs.aaai.org/index.php/AAAI/article/view/5747) | AAAI'20 | Add/Remove Edges | GCN, GAT, HG, GraphSAGE, HO | NC | [Available](https://github.com/victorchen96/MadGap/tree/master)
| Gatk | [Nettack](https://dl.acm.org/doi/abs/10.1145/3219819.3220078) | KDD'18 | Add/Remove Edges | GCN | NC | [Available](https://github.com/danielzuegner/nettack)
| Gatk | [Metattack](https://arxiv.org/abs/1902.08412) | ICLR'19 | Add/Remove Edges | GCN | NC | [Available](https://github.com/danielzuegner/gnn-meta-attack)
| Gatk | [LinLBP](https://dl.acm.org/doi/abs/10.1145/3319535.3354206) | CCS'19 | Add/Remove Edges | GCN | NC | -
| Gatk | [Black-box Attack](https://arxiv.org/abs/2108.09513) | CCS'21 | Add/Remove Edges | GIN, SAG, GUnet | GC | [Available](https://github.com/mujm/CCS21_GNNattack)
| Gatk | [Topo-attack](https://ieeexplore.ieee.org/document/9046288) | IJCAI'19 | Add/Remove Edges | GCN | NC | [Available](https://github.com/KaidiXu/GCN_ADV_Train)
| Gatk | [GF-attack](https://arxiv.org/abs/1908.01297) | AAAI'20 | Add/Remove Edges | GCN, SGC | NC | [Available](https://github.com/SwiftieH/GFAttack)
| Gatk | [LowBlow](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) | WSDM'20 | Add/Remove Edges | GCN | NC | -
| Gatk | [GUA](https://arxiv.org/abs/2002.04784) | IJCAI'21 | Add/Remove Edges | GCN, GAT | NC | [Available](https://github.com/chisam0217/Graph-Universal-Attack)
| Gatk | [NE-attack](https://arxiv.org/abs/1809.01093) | ICML'19 | Add/Remove Edges | GCN | NC & LP | [Available](https://github.com/abojchevski/node_embedding_attack)
| Gatk | [Viking](https://arxiv.org/abs/2102.07164) | PAKDD'21 | Add/Remove Edges | GCN | NC & LP | [Available](https://github.com/virresh/viking)
| Gatk | [RL-attack](https://arxiv.org/abs/1806.02371) | ICML'18 | Add/Remove Edges | GCN | NC & GC | [Available](https://github.com/Hanjun-Dai/graph_adversarial_attack)




</div>

## Requirements
* `Python 3.8`
* `tensorflow-gpu 2.4.0`
* `cuda 11.0`
* `DGL`
* `numpy 1.20.3`
* `pytorch 1.1.3 (only if you run SGC, GraphSage, and DAGNN)`

## Experimental Devices
| Platform | Configuration |
| ---------- |---------- |
| CPU | 32-core Intel Xeon Platinum 8350C CPU (2.60GHz) |
| GPU | NVIDIA A100 SXM 80GB |

## Implementation

Please use the following scripts to execute either solution of EPD.

EPD Solution I:

An example of augmentation: Make augmentation by removing 100 heterophilic edges on cora
```
python EPDS1.py -dataset cora -heteE 100 -homoE 0 -type aug
```
EPD Solution I: 

An example of attack: Inject Attack by adding 100 heterophilic edges on cora
```
python EPDS1.py -dataset cora -heteE 100 -homoE 0 -type atk
```
EPD Solution II:

An example of augmentation: Make augmentation after applying the target-guided modification to cora
```
python EPDS2.py -heteE 100 -homoE 0 -dataset cora -epochs 10 -rn 1 -type aug 
```
EPD Solution II:

An example of attack: Make attack after applying the target-guided modification to cora
```
python EPDS2.py -heteE 100 -homoE 0 -dataset cora -type atk
```

After making perturbations on graphs, you can test the effectiveness of EPD on four widely used GNN backbones. Please follow the corresponding scripts:
```
GCN: python train.py
GraphSage: python graphsage.py
SGC: python sgc.py --dataset cora --gpu 0
DAGNN: python dagnn.py --dataset Cora --gpu 0 --runs 10 --lamb 0.005 --k 12
```
