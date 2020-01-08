# hsnw


[![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/hnsw.svg
[cl]: https://crates.io/crates/hnsw/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/hnsw/badge.svg
[dl]: https://docs.rs/hnsw/

[lo]: https://tokei.rs/b1/github/rust-photogrammetry/hnsw?category=code

Hierarchical Navigable Small World Graph for fast ANN search

Enable the `serde-impl` feature to serialize and deserialize `Hamming` and `Euclidean` types.

## Example

### Binary feature search using hamming distance

```rust
use hnsw::{Hamming, Searcher, HNSW};

fn main() {
    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Hamming<u128>> = HNSW::new();

    let features = [
        0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1100, 0b1001,
    ];

    // Insert all features. A searcher data structure is used to avoid performing
    // memory allocations every insertion and search. Reuse searchers for speed.
    for &feature in &features {
        hnsw.insert(Hamming(feature), &mut searcher);
    }

    // Allocate an array to store nearest neighbors.
    let mut neighbors = [!0; 8];
    // Pass the whole neighbors array as a slice. It will attempt to fill the whole array
    // with nearest neighbors from nearest to furthest.
    hnsw.nearest(&Hamming(0b0001), 24, &mut searcher, &mut neighbors);
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(&neighbors, &[0, 4, 7, 1, 2, 3, 5, 6]);
}
```

Distance is implemented for up to `Hamming<[u8x64; 32]>`, 16384 bits, and it utilizes SIMD for speed so
long as you use `RUSTFLAGS="-Ctarget-cpu=native" cargo build --release`. There are also impls
for `Hamming<u8>` through `Hamming<u128>`. You can impl the `Distance` trait on your own types,
including `Hamming<N>` where `N` is your own type, as that doesn't violate orphan rules.

If you want to determine the number of bytes at runtime, you can use the relatively inefficient, but
dynamic, impl of `Distance` for `Hamming<&[u8]>`. PRs that improve the performance of this impl are
appreciated!

### Floating-point search using euclidean distance

```rust
use hnsw::{Euclidean, Searcher, HNSW};
use packed_simd::f32x4;

fn main() {
    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Euclidean<f32x4>> = HNSW::new();

    let features = [
        f32x4::new(0.0, 0.0, 0.0, 1.0),
        f32x4::new(0.0, 0.0, 1.0, 0.0),
        f32x4::new(0.0, 1.0, 0.0, 0.0),
        f32x4::new(1.0, 0.0, 0.0, 0.0),
        f32x4::new(0.0, 0.0, 1.0, 1.0),
        f32x4::new(0.0, 1.0, 1.0, 0.0),
        f32x4::new(1.0, 1.0, 0.0, 0.0),
        f32x4::new(1.0, 0.0, 0.0, 1.0),
    ];

    for &feature in &features {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }

    // Allocate an array to store nearest neighbors.
    let mut neighbors = [!0; 8];
    // Pass the whole neighbors array as a slice. It will attempt to fill the whole array
    // with nearest neighbors from nearest to furthest.
    hnsw.nearest(
        &Euclidean(f32x4::new(0.0, 0.0, 0.0, 1.0)),
        24,
        &mut searcher,
        &mut neighbors,
    );
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(&neighbors, &[0, 4, 7, 1, 2, 3, 5, 6]);
}
```

`FloatingDistance` is implemented for up to `Euclidean<[f32x16; 256]>`, 4096 floats, and it utilizes
SIMD for speed so long as you use `RUSTFLAGS="-Ctarget-cpu=native" cargo build --release`. There are
also impls for `Euclidean<f32x2>` through `Euclidean<f32x16>`. You can impl the `FloatingDistance` trait
on your own types, including `Euclidean<N>` where `N` is your own type, as that doesn't violate orphan rules.

If you want to determine the number of floats at runtime, you can use the relatively inefficient, but
dynamic, impl of `FloatingDistance` for `Euclidean<&[f32]>`. PRs that improve the performance of this impl
are appreciated!

## Benchmarks

Here is a recall graph that you can [compare to its alternatives](http://ann-benchmarks.com/sift-256-hamming_10_hamming.html):

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_10000_m24.svg)

For more benchmarks and how to benchmark, see [`benchmarks.md`](./benchmarks.md).

## Implementation

This is based on the paper ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/pdf/1603.09320.pdf) by Yu. A. Malkov and D. A. Yashunin. This paper builds on the [original paper for NSW](http://www.iiis.org/CDs2011/CD2011IDI/ICTA_2011/PapersPdf/CT175ON.pdf). There are multiple papers written by the authors on NSW, of which that is the last and most up-to-date.

For more details about parameters and details of the implementation, see [`implementation.md`](./implementation.md).

## Credit

This is in no way a direct copy or reimplementation of [the original implementation](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h). This was made purely based on [the paper](https://arxiv.org/pdf/1603.09320.pdf) without reference to the original headers. The paper is very well written and easy to understand, with some minor exceptions, so I never needed to refer to the original headers as I thought I would when I began working on this. Thank you to the authors for your valuble contribution.

## Questions? Contributions? Excited?

Please visit the [Rust Photogrammetry Discord](https://discord.gg/d32jaam).

