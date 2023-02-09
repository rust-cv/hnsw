use hnsw::{
    metric::{EncodableFloat, Neighbor, SimpleEuclidean},
    Hnsw, Searcher,
};
use rand_pcg::Pcg64;

#[test]
fn random() {
    const SAMPLES: usize = 100;

    let mut searcher = Searcher::default();

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
            .take(SAMPLES)
            .collect(),
            query.to_vec(),
        )
    };

    let mut hnsw = Hnsw::<_, Vec<f32>, Pcg64, 12, 24>::new(SimpleEuclidean);

    for feature in features.clone() {
        hnsw.insert(feature, &mut searcher);
    }

    let neighbors = {
        let mut neighbors = [Neighbor {
            index: 0,
            distance: EncodableFloat { value: f32::MAX },
        }; SAMPLES];

        hnsw.nearest(&query, 24, &mut searcher, &mut neighbors);

        neighbors
    };

    let features: Vec<_> = {
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

    // neighbors.sort_by(|a, b| {
    //     a.distance
    //         .value
    //         .partial_cmp(&b.distance.value)
    //         .unwrap()
    //         .then(a.index.cmp(&b.index))
    // });
    // features.sort_by(|a, b| a.distance.value.partial_cmp(&b.distance.value).unwrap());

    println!(
        "{} {}",
        neighbors
            .iter()
            .filter(|neighbor| features.contains(neighbor))
            .count(),
        neighbors
            .iter()
            .filter(|neighbor| features.contains(neighbor))
            .count() as f32
            / features.len() as f32
    );

    assert!(
        neighbors
            .iter()
            .filter(|neighbor| features.contains(neighbor))
            .count() as f32
            / features.len() as f32
            >= 0.9
    );
}
