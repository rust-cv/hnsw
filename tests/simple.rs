use hnsw::*;

#[test]
fn insertion() {
    let mut searcher = Searcher::default();
    let mut hnsw = HNSW::new();

    let features = [
        0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0110, 0b1100, 0b1001,
    ];

    for &feature in &features {
        hnsw.insert(feature, &mut searcher);
    }
}
