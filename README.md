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

## Example

### Binary feature search using hamming distance

```rust
use hnsw::{Searcher, HNSW};
use space::{Hamming, Neighbor};

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
    let mut neighbors = [Neighbor::invalid(); 8];
    // Pass the whole neighbors array as a slice. It will attempt to fill the whole array
    // with nearest neighbors from nearest to furthest.
    hnsw.nearest(&Hamming(0b0001), 24, &mut searcher, &mut neighbors);
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(
        neighbors,
        [
            Neighbor {
                index: 0,
                distance: 0
            },
            Neighbor {
                index: 4,
                distance: 1
            },
            Neighbor {
                index: 7,
                distance: 1
            },
            Neighbor {
                index: 1,
                distance: 2
            },
            Neighbor {
                index: 2,
                distance: 2
            },
            Neighbor {
                index: 3,
                distance: 2
            },
            Neighbor {
                index: 5,
                distance: 3
            },
            Neighbor {
                index: 6,
                distance: 3
            }
        ]
    );
}
```

Please refer to the [`space` documentation](https://docs.rs/space/) for the trait and types regarding distance. It also contains special `Bits128` - `Bits4096` tuple structs that wrap an array of bytes and enable SIMD capability. Benchmarks provided use these SIMD impls.

### Floating-point search using euclidean distance

An implementation is currently not provided for euclidean distance after a recent refactor. Hamming distance was more relevant at the time, and so that was prioritized. To implement euclidean distance, do something roughly like the following:

```rust
struct Euclidean<'a>(&'a [f32]);

impl MetricPoint for Euclidean<'_> {
    fn distance(&self, rhs: &Self) -> u32 {
        space::f32_metric(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt(),
        )
    }
}
```

Note that the above implementation may have some numerical error on high dimensionality. In that case use a [Kahan sum](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) instead.

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

