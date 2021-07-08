//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{MetricPoint, Neighbor};

struct Euclidean(&'static [f64]);

impl MetricPoint for Euclidean {
    type Metric = u64;
    fn distance(&self, rhs: &Self) -> u64 {
        self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            .to_bits()
    }
}

fn test_hnsw() -> (Hnsw<Euclidean, Pcg64, 12, 24>, Searcher<u64>) {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new();

    let features = [
        &[0.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0, 1.0, 0.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 0.0],
        &[0.0, 0.0, 1.0, 1.0],
        &[0.0, 1.0, 1.0, 0.0],
        &[1.0, 1.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
    ];

    for &feature in &features {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }

    (hnsw, searcher)
}

#[test]
fn insertion() {
    test_hnsw();
}

#[test]
fn nearest_neighbor() {
    let (hnsw, mut searcher) = test_hnsw();
    let searcher = &mut searcher;
    let mut neighbors = [Neighbor {
        index: !0,
        distance: !0,
    }; 8];

    hnsw.nearest(
        &Euclidean(&[0.0, 0.0, 0.0, 1.0]),
        24,
        searcher,
        &mut neighbors,
    );
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
                distance: 4607182418800017408
            },
            Neighbor {
                index: 7,
                distance: 4607182418800017408
            },
            Neighbor {
                index: 1,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 2,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 3,
                distance: 4609047870845172685
            },
            Neighbor {
                index: 5,
                distance: 4610479282544200874
            },
            Neighbor {
                index: 6,
                distance: 4610479282544200874
            }
        ]
    );
}
