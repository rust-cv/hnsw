//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::*;

fn test_hnsw_discrete() -> (DiscreteHNSW<Hamming<u128>>, DiscreteSearcher<Hamming<u128>>) {
    let mut searcher = DiscreteSearcher::default();
    let mut hnsw = DiscreteHNSW::new();

    let features = [
        0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1100, 0b1001,
    ];

    for &feature in &features {
        hnsw.insert(Hamming(feature), &mut searcher);
    }

    (hnsw, searcher)
}

#[test]
fn insertion_discrete() {
    test_hnsw_discrete();
}

#[test]
fn nearest_neighbor_discrete() {
    let (hnsw, mut searcher) = test_hnsw_discrete();
    let searcher = &mut searcher;
    let mut neighbors = [!0; 8];

    hnsw.nearest(&Hamming(0b0001), 24, searcher, &mut neighbors);
    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();
    assert_eq!(&neighbors, &[0, 4, 7, 1, 2, 3, 5, 6]);
}
