//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{MetricPoint, Neighbor};

struct Euclidean(&'static [f64]);

impl MetricPoint for Euclidean {
    fn distance(&self, rhs: &Self) -> u64 {
        space::f64_metric(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
        )
    }
}

fn test_hnsw() -> (Hnsw<Euclidean, Pcg64, 12, 24>, Searcher) {
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
    let mut neighbors = [Neighbor::invalid(); 8];

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
                distance: 1072693248
            },
            Neighbor {
                index: 7,
                distance: 1072693248
            },
            Neighbor {
                index: 1,
                distance: 1073127582
            },
            Neighbor {
                index: 2,
                distance: 1073127582
            },
            Neighbor {
                index: 3,
                distance: 1073127582
            },
            Neighbor {
                index: 5,
                distance: 1073460858
            },
            Neighbor {
                index: 6,
                distance: 1073460858
            }
        ]
    );
}
