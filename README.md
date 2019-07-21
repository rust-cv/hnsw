# hsnw

Hierarchical Navigable Small World Graph for fast ANN search

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, of which that is the last and most up-to-date.

HNSW is an algorithm that creates layers of NSW graphs where the top layer is least refined and the "zero layer" is the most refined. The graph nodes are items from the search set in all cases and `M` edges are chosen by finding the `M` nearest-neighbors according to the graph's ANN search. The authors of HSNW determine a heuristic to select which items from the search set to include in each graph layer to maintain global connectivity. They also define some parameters, including `Mmax0`, `Mmax`, and `mL`. `Mmax0` is the maximum `M` for the "zero" layer, while `Mmax` is the maximum `M` for every other layer. `mL` is the total number of layers above the zero layer.
