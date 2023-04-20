use hnsw::{
    metric::{Neighbor, SimpleEuclidean},
    Hnsw,
};
use rand_pcg::Pcg64;

#[test]
fn random() {
    // const PUT_SAMPLES: usize = 10_000;
    let start = std::time::Instant::now();
    const PUT_SAMPLES: usize = 5_000;
    const TAKE_NEIGHBORS: usize = 50;

    let (features, query): (Vec<_>, _) = {
        use rand::Rng as _;
        let mut rng = rand::thread_rng();

        let mut query = [0f32; 4];
        rng.fill(&mut query);

        (
            std::iter::repeat_with(|| {
                let mut feature = [0f32; 4];
                rng.fill(&mut feature);

                feature.to_vec()
            })
            .take(PUT_SAMPLES)
            .collect(),
            query.to_vec(),
        )
    };

    let neighbors: Vec<_> = {
        let mut hnsw = Hnsw::<_, Vec<f32>, Pcg64, _, 12, 24>::new(
            SimpleEuclidean,
            hnsw::storage::NodeDB::new("/tmp/test.db"),
            // hnsw::storage::HashMap::new(),
        );
        let mut id: usize = 0;
        for feature in features.clone() {
            hnsw.insert(feature, id);
            id += 1;
        }

        hnsw.nearest(&query, 24, TAKE_NEIGHBORS)
    };

    let expect_neighbors: Vec<_> = {
        use hnsw::metric::Metric as _;

        let euclidean_distance = SimpleEuclidean;

        let mut features: Vec<_> = features
            .iter()
            .enumerate()
            .map(|(index, feature)| Neighbor {
                index,
                distance: euclidean_distance.distance(&query, feature),
            })
            .collect();

        features.sort_by(|a, b| a.distance.value.partial_cmp(&b.distance.value).unwrap());
        features.drain(0..TAKE_NEIGHBORS * 2).collect()
    };

    let matches = neighbors
        .iter()
        .filter(|neighbor| expect_neighbors.contains(neighbor))
        .count();

    println!(
        "features: {}, neighbors: {}, matches: {matches}",
        features.len(),
        neighbors.len(),
    );

    assert!(matches as f32 / neighbors.len() as f32 >= 0.9);

    println!("{:?}", start.elapsed());
}
