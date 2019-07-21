# hsnw

Hierarchical Navigable Small World Graph for fast ANN search

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin.

HNSW is an algorithm that creates layers of graphs where the root layer is least refined and the final layer is the most refined. The graph nodes are items from the search space in all cases and `M` edges are chosen by finding the `M` nearest-neighbors according to the graph's ANN search.
