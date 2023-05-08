use hnsw::{
    metric::{Neighbor, SimpleEuclidean},
    Hnsw,
};
use itertools::Itertools;
use rand_pcg::Pcg64;

#[test]
fn random() {
    const PUT_SAMPLES: usize = 5_000;
    let start = std::time::Instant::now();
    //const PUT_SAMPLES: usize = 5_000;
    const QUERY_SAMPLES: usize = 50;

    let (features, query): (Vec<_>, Vec<_>) = {
        use rand::Rng as _;

        let repeater = || {
            let mut rng = rand::thread_rng();
            let mut f = [0f32; 4];
            rng.fill(&mut f);
            f.to_vec()
        };

        (
            std::iter::repeat_with(&repeater)
                .take(PUT_SAMPLES)
                .collect(),
            std::iter::repeat_with(repeater)
                .take(QUERY_SAMPLES)
                .collect(),
        )
    };

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

    let mut pass = 0;

    for q in query.clone() {
        use hnsw::metric::Metric as _;
        let euclidean_distance = SimpleEuclidean;

        let expected = features
            .iter()
            .enumerate()
            .min_by_key(|(_, feature)| euclidean_distance.distance(&q, feature))
            .unwrap();
        let found = hnsw.nearest(&q, 24, 24);
        if expected.0 == found[0].index {
            pass += 1;
        }
    }

    println!("pass {} / {} tests", pass, query.len());
    assert!(pass >= 10);

    println!("{:?}", start.elapsed());
}

#[test]
fn random2() {
    const PUT_SAMPLES: usize = 5_000;
    let start = std::time::Instant::now();
    //const PUT_SAMPLES: usize = 5_000;
    const QUERY_SAMPLES: usize = 50;

    let (features, query): (Vec<_>, Vec<_>) = {
        use rand::Rng as _;

        let repeater = || {
            let mut rng = rand::thread_rng();
            let mut f = [0f32; 4];
            rng.fill(&mut f);
            f.to_vec()
        };

        (
            std::iter::repeat_with(&repeater)
                .take(PUT_SAMPLES)
                .collect(),
            std::iter::repeat_with(repeater)
                .take(QUERY_SAMPLES)
                .collect(),
        )
    };

    let mut hnsw = Hnsw::<_, Vec<f32>, Pcg64, _, 12, 24>::new(
        SimpleEuclidean,
        hnsw::storage::NodeDB::new("/tmp/test2.db"),
        // hnsw::storage::HashMap::new(),
    );
    let mut id: usize = 0;
    for feature in features.clone() {
        hnsw.insert(feature, id);
        id += 1;
    }

    let mut pass = 0;

    for q in query.clone() {
        use hnsw::metric::Metric as _;
        let euclidean_distance = SimpleEuclidean;

        let mut expected = features
            .iter()
            .enumerate()
            .map(|(index, feature)| Neighbor {
                index,
                distance: euclidean_distance.distance(&q, feature),
            })
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| a.distance.cmp(&b.distance));
        let top10 = expected.drain(0..10).collect::<Vec<_>>();
        let found = hnsw.nearest(&q, 24, 24);
        if top10.contains(&found[0]) {
            pass += 1;
        }
    }

    println!("pass {} / {} tests", pass, query.len());
    assert!(pass as f32 / query.len() as f32 >= 0.8);

    println!("{:?}", start.elapsed());
}
