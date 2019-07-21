# hsnw

Hierarchical Navigable Small World Graph for fast ANN search

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, of which that is the last and most up-to-date.

HNSW is an algorithm that creates layers of NSW graphs where the top layer is least refined and the "zero layer" is the most refined. The graph nodes are items from the search set in all cases and `M` edges are chosen by finding the `M` nearest-neighbors according to the graph's ANN search. The authors of HSNW probabalistically select which items from the search set to include in each graph layer and below to maintain global connectivity. This is effectively combining the concept of skip lists with NSW.

The authors define some parameters:

- `M` is the number of nearest-neighbors to connect a new entry to when it is inserted.
    - Choose this depending on the dimensionality and recall rate.
    - The paper provides no formula, just some graphs and guidance, so this number might need to be fine-tuned.
    - For 5 million 128-dimensional SIFT floating-point features, this value was found to be optimal up to recalls of around 0.95.
    - The paper suggests an `M` between 5 and 48.
    - Benchmark your dataset to find the optimal `M`.
- `Mmax0` is the maximum `M` for the "zero" layer.
    - It should be set to `M * 2`, which the paper claims is relatively empirically optimal, but makes no mathematical claim.
        - See the graphs in figure 6 to see the empirical evidence.
- `Mmax` is the maximum `M` for every other layer.
    - Just set this to `M`. The paper doesn't say what to do, but the code [does this](https://github.com/nmslib/hnswlib/blob/3c13bc9d18e8242815c95b76b40103b92a65619c/hnswlib/hnswalg.h#L36).
- `mL` is a parameter the controls the random selection of the max layer an insertion will appear on.
    - This parameter is chosen to probabalistically approximate a skip list with an average of one element of overlap between layers.
    - The formula for the maximum layer `l` that an insertion appears on is `-ln(unif(0..1)) * mL`.
    - `mL` is chosen to be `1 / ln(M)`.
    - The chosen `mL` seems to empirically be ideal according to the paper's findings.
- `ef` is the number of nearest neighbors to keep in a priority queue while searching.
    - This is the parameter controlled to change the recall rate.
    - In their code they [hardcode this to `10`](https://github.com/nmslib/hnswlib/blob/3c13bc9d18e8242815c95b76b40103b92a65619c/hnswlib/hnswalg.h#L39) for some reason.
- `efConstruction` is the `ef` to use when searching for nearest neighbors when inserting.
    - This parameter is different from `ef` so that the quality of the graph can be improved (closer to the Delaunay graph)
        - Improving the quality of the graph means faster query time at the same recall, but slower graph construction
    - This may need to be fine-tuned for your needs.
        - The authors used a value of `100` on a 10 million SIFT dataset to get good quality.
        - A value of `40` on a 200 million SIFT dataset still gets virtually the same results as `500`.
    - Increasing this beyond a certian point does practically nothing at a high cost.
    - See figure 10 for some data.

## Credit

This is in no way a direct copy or reimplementation of [the original implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h), but that implementation will be reffered to in the design of this implementation whenever necessary as it has an Apache license. Thanks to the original authors and their choice to use a permissive license =).


