# hsnw

Hierarchical Navigable Small World Graph for fast ANN search

## Benchmarks

Here is a recall graph that you can [compare to its alternatives](http://ann-benchmarks.com/sift-256-hamming_10_hamming.html):

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_10000_m24.svg)

The above benchmark shows the speed boost this gives for 10-NN, but this datastructure is intended (by me) to be used for fast feature matching in images. When matching features in images, they must satisfy the lowe's ratio, which means that the first match must be significantly better than the second match or there may be ambiguity. For that purpose, here are several 2-NN benchmarks for 256-bit binary descriptors:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_10000_m12.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_100000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_1000000_m24.svg)

Please note that `M = 24` is not optimal for the above graph at 10000 items. The default `M` on this implementation is `12`, which can be fairly optimal for the 2-NN case.

You might want to know the performance of HNSW and how it changes when you increase the number of nearest neighbors. Here are some graphs of some 50-NN searches with different `M`:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m12.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m52.svg)

Finally, here are some more 10-NN benchmarks to satisfy your curiosity:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_100000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_1000000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_1000000_m48.svg)

You can find benchmarks of HNSW with 256-bit binary features vs linear search on 2-NN with `ef` parameter of `24`, `M` parameter of `24` (very high recall), and `efConstruction` set to `400` [here](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/report/index.html). This compares it against linear search, which is pretty fast in small datasets. This is not really a valid comparison because it is comparing apples to oranges (a linear search which is perfect with an ANN algorithm that is getting worse at recall). However, this benchmark is useful for profiling the code, so I will share its results here. Please use the recall graphs above as the actual point of comparison.

You can also generate recall graphs (native cpu instructions recommended). Use the following to see how:

```bash
# Discrete version (binary descriptors)
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example recall_discrete -- --help
# Non-discrete version (floating descriptors)
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example recall -- --help
```

If you don't want to use random data (highly recommended), please use the `generate.py` script like so:

```bash
# AKAZE binary descriptors
python3 scripts/generate.py akaze 2011_09_29/2011_09_29_drive_0071_extract/image_00/data/*.png > data/akaze
# KAZE floating point descriptors
python3 scripts/generate.py kaze 2011_09_29/2011_09_29_drive_0071_extract/image_00/data/*.png > data/kaze
```

This code ran againt the Kitti dataset ` 2011_09_29_drive_0071`. You can find that [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). Download "[unsynced+unrectified data]". It is a large (4.1 GB) download, so it may take some time.

You can still run the above generation against any dataset you would like if you would like to test its performance on said dataset.

This crate may take a while to compile due to the use of `typenum` and `generic-array`. If you dislike this, consider contributing to some issues labeled [A-const-generics](https://github.com/rust-lang/rust/labels/A-const-generics) in Rust to help push along the const generics support in the compiler. The `recall_discrete` generator is especially time-consuming to build.

## Implementation

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, of which that is the last and most up-to-date.

HNSW is an algorithm that creates layers of NSW graphs where the top layer is least refined and the "zero layer" is the most refined. The graph nodes are items from the search set in all cases and `M` edges are chosen by finding the `M` nearest-neighbors according to the graph's ANN search. The authors of HSNW probabalistically select which items from the search set to include in each graph layer and below to maintain global connectivity. This is effectively combining the concept of skip lists with NSW.

The authors define some parameters:

- `M` is the number of nearest-neighbors to connect a new entry to when it is inserted.
    - Choose this depending on the dimensionality and recall rate.
    - The paper provides no formula, just some graphs and guidance, so this number might need to be fine-tuned.
    - The paper suggests an `M` between 5 and 48.
    - Benchmark your dataset to find the optimal `M`.
- `Mmax0` is the maximum `M` for the "zero" layer.
    - It should be set to `M * 2`, which the paper claims is relatively empirically optimal, but makes no mathematical claim.
        - See the graphs in figure 6 to see the empirical evidence.
    - In the code of this crate, this is the parameter `M0`.
- `Mmax` is the maximum `M` for every non-zero layer.
    - In the code of this crate, this is the parameter `M`.
- `mL` is a parameter the controls the random selection of the max layer an insertion will appear on.
    - This parameter is chosen to probabalistically approximate a skip list with an average of one element of overlap between layers.
    - The formula for the maximum layer `l` that an insertion appears on is `-ln(unif(0..1)) * mL`.
    - `mL` is chosen to be `1 / ln(M)`.
    - The chosen `mL` seems to empirically be ideal according to the paper's findings, so this is not exposed to the user.
- `ef` is the number of nearest neighbors to keep in a priority queue while searching.
    - This is the parameter controlled to change the recall rate.
    - This is also reffered to as `ef` in this crate.
- `efConstruction` is the `ef` to use when searching for nearest neighbors when inserting.
    - This parameter is different from `ef` so that the quality of the graph can be improved (closer to the Delaunay graph).
        - Improving the quality of the graph means faster query time at the same recall, but slower graph construction.
    - This may need to be fine-tuned for your needs.
        - The authors used a value of `100` on a 10 million SIFT dataset to get good quality.
        - A value of `40` on a 200 million SIFT dataset still gets virtually the same results as `500`.
    - Increasing this beyond a certian point does practically nothing at a high cost.
    - See figure 10 for some data.
    - I would set this to about `100` if insertion performance is not a concern.
        - If insertion performance is a concern, benchmark it on your dataset.

## Credit

This is in no way a direct copy or reimplementation of [the original implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). This was made purely based on [the paper](https://arxiv.org/pdf/1603.09320.pdf) without reference to the original headers. The paper is very well written and easy to understand, with some minor exceptions, so I never needed to refer to the original headers as I thought I would when I began working on this. Thank you to the authors for your valuble contribution.


