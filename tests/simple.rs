//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{Searcher, HNSW};
use space::MetricPoint;

struct Euclidean(&'static [f32]);

impl MetricPoint for Euclidean {
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

fn test_hnsw() -> (HNSW<Euclidean>, Searcher) {
    let mut searcher = Searcher::default();
    let mut hnsw = HNSW::new();

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
    let mut neighbors = [!0; 8];

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
    assert_eq!(&neighbors, &[0, 4, 7, 1, 2, 3, 5, 6]);
}
