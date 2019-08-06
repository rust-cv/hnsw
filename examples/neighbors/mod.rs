use criterion::*;
use hnsw::*;
use rand::distributions::{Bernoulli, Standard};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::rc::Rc;

/// This is the probability each bit of an inlier will be different.
/// This comes from "Online Nearest Neighbor Search in Hamming Space"
/// in figure 2a, where 1-NN has an average search radius of 11 for
/// 128-bit features. I make the assumption here that the inliers
/// exist on a binomial distribution over 128 choices centered at
/// 11, which is consistent with the inlier statistics found in the paper
/// "ORB: an efficient alternative to SIFT or SURF".
const BIT_DIFF_PROBABILITY_OF_INLIER: f64 = 0.0859;

fn bench_neighbors(c: &mut Criterion) {
    let space_mags = 0..=22;
    let all_sizes = (space_mags).map(|n| 2usize.pow(n));
    let rng = Pcg64::from_seed([5; 32]);
    // Get the bigest input size and then generate all inputs from that.
    eprintln!("Generating random inputs...");
    let all_input = rng
        .sample_iter(&Standard)
        .take(all_sizes.clone().rev().next().unwrap())
        .collect::<Vec<u128>>();
    let mut rng = Pcg64::from_seed([6; 32]);
    eprintln!("Done.");
    eprintln!("Generating HNSWs...");
    let bernoulli = Bernoulli::new(BIT_DIFF_PROBABILITY_OF_INLIER).unwrap();
    let hnsw_map = Rc::new(HashMap::<_, _>::from_iter(all_sizes.clone().map(|total| {
        eprintln!("Generating HNSW size {}...", total);
        let range = 0..total;
        let mut hnsw: HNSW = HNSW::new();
        let mut searcher = Searcher::default();
        for i in range.clone() {
            hnsw.insert(all_input[i], &mut searcher);
        }
        // In the paper they choose 1000 samples that arent in the data set.
        let inliers: Vec<u128> = all_input[0..total]
            .choose_multiple(&mut rng, 1000)
            .map(|&feature| {
                let mut feature = feature;
                for bit in 0..128 {
                    let choice: bool = rng.sample(&bernoulli);
                    feature ^= (choice as u128) << bit;
                }
                feature
            })
            .collect();
        (total, (hnsw, inliers))
    })));
    eprintln!("Done.");
    c.bench(
        "neighbors",
        ParameterizedBenchmark::new(
            "nearest_1_hnsw",
            {
                let hnsw_map = hnsw_map.clone();
                move |bencher: &mut Bencher, total: &usize| {
                    let (hnsw, inliers) = &hnsw_map[total];
                    let mut cycle_range = inliers.iter().cloned().cycle();
                    let mut searcher = Searcher::default();
                    bencher.iter(|| {
                        let feature = cycle_range.next().unwrap();
                        let mut neighbors = [0; 1];
                        hnsw.nearest(feature, 24, &mut searcher, &mut neighbors)
                            .len()
                    });
                }
            },
            all_sizes,
        )
        .with_function("nearest_1_linear", {
            let hnsw_map = hnsw_map.clone();
            move |bencher: &mut Bencher, &total: &usize| {
                let (_, inliers) = &hnsw_map[&total];
                let mut cycle_range = inliers.iter().cloned().cycle();
                bencher.iter(|| {
                    let feature = cycle_range.next().unwrap();
                    all_input[0..total]
                        .iter()
                        .cloned()
                        .min_by_key(|n| (feature ^ n).count_ones())
                });
            }
        })
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic)),
    );
}

fn config() -> Criterion {
    Criterion::default().sample_size(32)
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_neighbors
}
