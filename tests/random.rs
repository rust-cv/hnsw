use hnsw::{
    metric::{EncodableFloat, Neighbor, SimpleEuclidean},
    Hnsw, Searcher,
};
use rand_pcg::Pcg64;

#[test]
fn random() {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::<_, Vec<f32>, Pcg64, 12, 24>::new(SimpleEuclidean);
    let features = [
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ];

    let query = vec![0.0, 0.0, 0.0, 1.0];

    for feature in features.clone() {
        hnsw.insert(feature, &mut searcher);
    }

    let mut neighbors = [Neighbor {
        index: 0,
        distance: EncodableFloat { value: f32::MAX },
    }; 8];
    hnsw.nearest(&query, 24, &mut searcher, &mut neighbors);

    let mut features: Vec<_> = {
        use hnsw::metric::Metric as _;

        let euclidean_distance = SimpleEuclidean;

        features
            .iter()
            .enumerate()
            .map(|(index, feature)| Neighbor {
                index,
                distance: euclidean_distance.distance(&query, feature),
            })
            .collect()
    };

    neighbors.sort_by(|a, b| {
        a.distance
            .value
            .partial_cmp(&b.distance.value)
            .unwrap()
            .then(a.index.cmp(&b.index))
    });
    features.sort_by(|a, b| a.distance.value.partial_cmp(&b.distance.value).unwrap());

    assert_eq!(neighbors, features[..]);
}
