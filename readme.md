# Inferring from References with Differences for Semi-Supervised Node Classification on Graphs

Following the application of Deep Learning to graphic data, Graph Neural Networks (GNNs) have become the dominant method for Node Classification on graphs in recent years. To assign nodes with preset labels, most GNNs inherit the end-to-end way of Deep Learning in which node features are input to models while labels of pre-classified nodes are used for supervised learning. However, while these methods can make full use of node features and their associations, they treat labels separately and ignore the structural information of those labels. To utilize information on label structures, this paper proposes a method called 3ference that infers from references with differences. Specifically, 3ference predicts what label a node has according to the features of that node in concatenation with both features and labels of its relevant nodes. With the additional information on labels of relevant nodes, 3ference captures the transition pattern of labels between nodes, as subsequent analysis and visualization revealed. Experiments on a synthetic graph and seven real-world graphs proved that this knowledge about label associations helps 3ference to predict accurately with fewer parameters, fewer pre-classified nodes, and varying label patterns compared with GNNs.

## State of the Art

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-cora)](https://paperswithcode.com/sota/node-classification-on-cora?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-citeseer)](https://paperswithcode.com/sota/node-classification-on-citeseer?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-pubmed)](https://paperswithcode.com/sota/node-classification-on-pubmed?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-amazon-computers-1)](https://paperswithcode.com/sota/node-classification-on-amazon-computers-1?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-amazon-photo-1)](https://paperswithcode.com/sota/node-classification-on-amazon-photo-1?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-coauthor-cs)](https://paperswithcode.com/sota/node-classification-on-coauthor-cs?p=inferring-from-references-with-differences)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/inferring-from-references-with-differences/node-classification-on-coauthor-physics)](https://paperswithcode.com/sota/node-classification-on-coauthor-physics?p=inferring-from-references-with-differences)

## Citation

```
@Article{math10081262,
AUTHOR = {Luo, Yi and Luo, Guangchun and Yan, Ke and Chen, Aiguo},
TITLE = {Inferring from References with Differences for Semi-Supervised Node Classification on Graphs},
JOURNAL = {Mathematics},
VOLUME = {10},
YEAR = {2022},
NUMBER = {8},
ARTICLE-NUMBER = {1262},
URL = {https://www.mdpi.com/2227-7390/10/8/1262},
ISSN = {2227-7390},
DOI = {10.3390/math10081262}
}
```
