# hnsw


[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/hnsw.svg
[cl]: https://crates.io/crates/hnsw/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/hnsw/badge.svg
[dl]: https://docs.rs/hnsw/

[lo]: https://tokei.rs/b1/github/rust-cv/hnsw?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

Hierarchical Navigable Small World Graph for fast ANN search

Enable the `serde` feature to serialize and deserialize `HNSW`.

## Tips

A good default for M and M0 parameters is 12 and 24 respectively. According to the paper, M0 should always be double M,
but you can change both of them freely.

## Example

To see how this might be used with hamming space, see `tests/simple_discrete.rs`. To see how this might be used with euclidean space, see `tests/simple.rs`.

Note that the euclidean implementation in the test may have some numerical errors and fail to implement the triangle inequality, especially on high dimensionality. Use a [Kahan sum](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) instead for proper usage. It also may not utilize SIMD, but using an array may help with that.

Please refer to the [`space` documentation](https://docs.rs/space/) for the trait and types regarding distance. It also contains special `Bits128` - `Bits4096` tuple structs that wrap an array of bytes and enable SIMD capability. Benchmarks provided use these SIMD impls.

## Benchmarks

Here is a recall graph that you can [compare to its alternatives](http://ann-benchmarks.com/sift-256-hamming_10_hamming.html):

![Recall Graph](http://vadixidav.github.io/hnsw/839611966a1550d5cba599c78002ee68311e4c37/nn10_256bit_10000_m24.svg)

For more benchmarks and how to benchmark, see [`benchmarks.md`](./benchmarks.md).

## Implementation

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, which preceeded HNSW.

For more details about parameters and details of the implementation, see [`implementation.md`](./implementation.md).

## Credit

This is in no way a direct copy or reimplementation of [the original implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). This was made purely based on [the paper](https://arxiv.org/pdf/1603.09320.pdf) without reference to the original headers. The paper is very well written and easy to understand, with some minor exceptions. Thank you to the authors for your valuble contribution.

## Questions? Contributions? Excited?

Please visit the [Rust CV Discord](https://discord.gg/d32jaam).

