//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{Euclidean, Searcher, HNSW};
use packed_simd::f32x4;

fn test_hnsw() -> (HNSW<Euclidean<f32x4>>, Searcher) {
    let mut searcher = Searcher::default();
    let mut hnsw = HNSW::new();

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
        &Euclidean(f32x4::new(0.0, 0.0, 0.0, 1.0)),
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
