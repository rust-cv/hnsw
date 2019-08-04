//! More robust tests that are generated from substantial random data.

use hnsw::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_core::SeedableRng;
use rand_distr::{Bernoulli, Standard};
use rand_pcg::Pcg64;

// This can be adjusted lower if it is too slow.
const SEARCH_SPACE_SIZE: usize = 1 << 16;

#[test]
fn linear_1_nn() {
    let mut searcher = Searcher::default();
    let searcher = &mut searcher;
    let mut hnsw: HNSW = HNSW::new();
    let mut output = [!0; 1];

    let prng = Pcg64::from_seed([5; 32]);
    let mut rngiter = prng.sample_iter(&Standard);
    let space = (&mut rngiter)
        .take(SEARCH_SPACE_SIZE)
        .collect::<Vec<u128>>();
    let search = (&mut rngiter).take(100).collect::<Vec<u128>>();

    for &feature in &space {
        hnsw.insert(feature, searcher);
    }

    let mut pass = 0;

    for &feature in &search {
        // Use linear search to find the nearest neighbor.
        let nearest = space
            .iter()
            .enumerate()
            .min_by_key(|(_, &space_feature)| (feature ^ space_feature).count_ones())
            .unwrap();
        // Use HNSW to find the nearest neighbor.
        hnsw.nearest(feature, 24, searcher, &mut output);
        // Get their respective found features.
        let linear = *nearest.1;
        let hnsw = space[output[0] as usize];
        eprintln!("{:0128b}", linear);
        eprintln!("{:0128b}", hnsw);
        eprintln!("linear distance: {}", (linear ^ feature).count_ones());
        eprintln!("hnsw distance: {}", (hnsw ^ feature).count_ones());
        if (linear ^ feature).count_ones() == (hnsw ^ feature).count_ones() {
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
    let mut hnsw: HNSW = HNSW::new();
    let mut output = [!0; 1];

    const BIT_DIFF_PROBABILITY_OF_INLIER: f64 = 0.0859;

    let prng = Pcg64::from_seed([5; 32]);
    let space = prng
        .sample_iter(&Standard)
        .take(SEARCH_SPACE_SIZE)
        .collect::<Vec<u128>>();
    let mut prng_elem_chooser = Pcg64::from_seed([6; 32]);
    let mut prng_bit_chooser = Pcg64::from_seed([7; 32]);
    let bernoulli = Bernoulli::new(BIT_DIFF_PROBABILITY_OF_INLIER).unwrap();
    let search = space
        .choose_multiple(&mut prng_elem_chooser, 100)
        .cloned()
        .map(|mut feature| {
            for bit in 0..128 {
                let choice: bool = prng_bit_chooser.sample(&bernoulli);
                feature ^= (choice as u128) << bit;
            }
            feature
        })
        .collect::<Vec<u128>>();

    for &feature in &space {
        hnsw.insert(feature, searcher);
    }

    let mut pass = 0;

    for &feature in &search {
        // Use linear search to find the nearest neighbor.
        let nearest = space
            .iter()
            .enumerate()
            .min_by_key(|(_, &space_feature)| (feature ^ space_feature).count_ones())
            .unwrap();
        // Use HNSW to find the nearest neighbor.
        hnsw.nearest(feature, 24, searcher, &mut output);
        // Get their respective found features.
        let linear = *nearest.1;
        let hnsw = space[output[0] as usize];
        eprintln!("{:0128b}", linear);
        eprintln!("{:0128b}", hnsw);
        eprintln!("linear distance: {}", (linear ^ feature).count_ones());
        eprintln!("hnsw distance: {}", (hnsw ^ feature).count_ones());
        if (linear ^ feature).count_ones() == (hnsw ^ feature).count_ones() {
            pass += 1;
        }
    }

    eprintln!("pass: {}/100", pass);
    assert!(pass >= 90);
}
