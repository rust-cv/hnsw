use hnsw::*;

fn test_hnsw() -> (HNSW, Searcher) {
    let mut searcher = Searcher::default();
    let mut hnsw = HNSW::new();

    let features = [
        0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1100, 0b1001,
    ];

    for &feature in &features {
        hnsw.insert(feature, &mut searcher);
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
    let mut neighbors = [!0; 3];

    hnsw.nearest(0b0001, searcher, &mut neighbors);
    neighbors.sort_unstable();
    assert_eq!(&neighbors, &[0, 4, 7]);
}
