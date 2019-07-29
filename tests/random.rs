//! More robust tests that are generated from substantial random data.

use hnsw::*;
use rand::Rng;
use rand_core::SeedableRng;
use rand_distr::Standard;
use rand_pcg::Pcg64;

#[test]
fn linear_1_nn() {
    let mut searcher = Searcher::default();
    let searcher = &mut searcher;
    let mut hnsw = HNSW::new();
    let mut output = [!0; 1];

    let prng = Pcg64::from_seed([5; 32]);
    let mut rngiter = prng.sample_iter(&Standard);
    let space = (&mut rngiter).take(1 << 16).collect::<Vec<u128>>();
    let search = (&mut rngiter).take(10).collect::<Vec<u128>>();

    for &feature in &space {
        hnsw.insert(feature, searcher);
    }

    for &feature in &search {
        // Use linear search to find the nearest neighbor.
        let nearest = space
            .iter()
            .enumerate()
            .min_by_key(|(_, &space_feature)| (feature ^ space_feature).count_ones())
            .unwrap();
        // Use HNSW to find the nearest neighbor.
        assert_eq!(
            &[nearest.0 as u32],
            hnsw.nearest(feature, searcher, &mut output)
        );
    }
}
