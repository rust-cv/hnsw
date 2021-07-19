//! More robust tests that are generated from substantial random data.

use bitarray::{BitArray, Hamming};
use hnsw::*;
use rand::distributions::{Bernoulli, Standard};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_core::SeedableRng;
use rand_pcg::Pcg64;
use space::Neighbor;

// This can be adjusted lower if it is too slow.
const SEARCH_SPACE_SIZE: usize = 1 << 10;

#[test]
fn linear_1_nn() {
    let mut searcher = Searcher::default();
    let searcher = &mut searcher;
    let mut hnsw: Hnsw<Hamming, BitArray<16>, Pcg64, 12, 24> = Hnsw::default();
    let mut output = [Neighbor {
        index: !0,
        distance: !0,
    }; 1];

    let prng = Pcg64::from_seed([5; 32]);
    let mut rngiter = prng.sample_iter(&Standard).map(BitArray::new);
    let space = (&mut rngiter).take(SEARCH_SPACE_SIZE).collect::<Vec<_>>();
    let search = (&mut rngiter).take(100).collect::<Vec<_>>();

    for &feature in &space {
        hnsw.insert(feature, searcher);
    }

    let mut pass = 0;

    for &feature in &search {
        // Use linear search to find the nearest neighbor.
        let nearest = space
            .iter()
            .enumerate()
            .min_by_key(|(_, &space_feature)| feature.distance(&space_feature))
            .unwrap();
        // Use HNSW to find the nearest neighbor.
        hnsw.nearest(&feature, 24, searcher, &mut output);
        // Get their respective found features.
        let linear = *nearest.1;
        let hnsw = space[output[0].index];
        eprintln!("{:?}", linear);
        eprintln!("{:?}", hnsw);
        eprintln!("linear distance: {}", linear.distance(&feature));
        eprintln!("hnsw distance: {}", hnsw.distance(&feature));
        if linear.distance(&feature) == hnsw.distance(&feature) {
            pass += 1;
        }
    }

    eprintln!("pass: {}/100", pass);
    assert!(pass >= 10);
}

/// Does the same thing as linear_1_nn, but purposefully generates inliers in the search set.
#[test]
fn linear_1_nn_inliers() {
    let mut searcher = Searcher::default();
    let searcher = &mut searcher;
    let mut hnsw: Hnsw<Hamming, BitArray<16>, Pcg64, 12, 24> = Hnsw::default();
    let mut output = [Neighbor {
        index: !0,
        distance: !0,
    }; 1];

    const BIT_DIFF_PROBABILITY_OF_INLIER: f64 = 0.0859;

    let prng = Pcg64::from_seed([5; 32]);
    let space = prng
        .sample_iter(&Standard)
        .map(BitArray::new)
        .take(SEARCH_SPACE_SIZE)
        .collect::<Vec<_>>();
    let mut prng_elem_chooser = Pcg64::from_seed([6; 32]);
    let mut prng_bit_chooser = Pcg64::from_seed([7; 32]);
    let bernoulli = Bernoulli::new(BIT_DIFF_PROBABILITY_OF_INLIER).unwrap();
    let search = space
        .choose_multiple(&mut prng_elem_chooser, 100)
        .cloned()
        .map(|mut feature| {
            for bit in 0..128 {
                let choice: bool = prng_bit_chooser.sample(&bernoulli);
                feature[bit / 8] ^= (choice as u8) << (bit % 8);
            }
            feature
        })
        .collect::<Vec<_>>();

    for &feature in &space {
        hnsw.insert(feature, searcher);
    }

    let mut pass = 0;

    for &feature in &search {
        // Use linear search to find the nearest neighbor.
        let nearest = space
            .iter()
            .enumerate()
            .min_by_key(|(_, &space_feature)| feature.distance(&space_feature))
            .unwrap();
        // Use HNSW to find the nearest neighbor.
        hnsw.nearest(&feature, 24, searcher, &mut output);
        // Get their respective found features.
        let linear = *nearest.1;
        let hnsw = space[output[0].index];
        eprintln!("{:?}", linear);
        eprintln!("{:?}", hnsw);
        eprintln!("linear distance: {}", linear.distance(&feature));
        eprintln!("hnsw distance: {}", hnsw.distance(&feature));
        if linear.distance(&feature) == hnsw.distance(&feature) {
            pass += 1;
        }
    }

    eprintln!("pass: {}/100", pass);
    assert!(pass >= 90);
}
