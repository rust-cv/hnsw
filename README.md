# HNSW DB
Embeddable integration of Hierarchical Navigable Small World(HNSW) Graph for ANN search.
This project is forked from [this repository](https://github.com/rust-cv/hnsw).
We aim to integrate HNSW and key-value datastore such as LevelDB for better memory usage and persistency reasons.

## Tips

A good default for M and M0 parameters is 12 and 24 respectively. According to the paper, M0 should always be double M,
but you can change both of them freely.

## Example
To see how this might be used with euclidean space, see `tests/simple.rs`.

## Algorithm
This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, which preceeded HNSW.

## Credit

This is in no way a direct copy or reimplementation of [the original implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). This was made purely based on [the paper](https://arxiv.org/pdf/1603.09320.pdf) without reference to the original headers. The paper is very well written and easy to understand, with some minor exceptions. Thank you to the authors for your valuble contribution.
