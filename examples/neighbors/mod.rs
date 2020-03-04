use criterion::*;
use hamming_heap::FixedHammingHeap;
use hnsw::*;
use space::*;
use std::collections::HashMap;
use std::io::Read;
use std::iter::FromIterator;
use std::rc::Rc;

fn bench_neighbors(c: &mut Criterion) {
    let space_mags = 0..=15;
    let all_sizes = (space_mags).map(|n| 2usize.pow(n));
    let filepath = "data/akaze";
    let total_descriptors = all_sizes.clone().rev().next().unwrap();
    let descriptor_size_bytes = 61;
    let total_query_strings = 10000;

    // Read in search space.
    eprintln!(
        "Reading {} search space descriptors of size {} bytes from file \"{}\"...",
        total_descriptors, descriptor_size_bytes, filepath
    );
    let mut file = std::fs::File::open(filepath).expect("unable to open file");
    let mut v = vec![0u8; total_descriptors * descriptor_size_bytes];
    file.read_exact(&mut v).expect(
        "unable to read enough search descriptors from the file; add more descriptors to file",
    );
    let search_space: Rc<Vec<Hamming<Bits256>>> = Rc::new(
        v.chunks_exact(descriptor_size_bytes)
            .map(|b| {
                let mut arr = [0; 32];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                Hamming(Bits256(arr))
            })
            .collect(),
    );
    eprintln!("Done.");

    // Read in query strings.
    eprintln!(
        "Reading {} query descriptors of size {} bytes from file \"{}\"...",
        total_query_strings, descriptor_size_bytes, filepath
    );
    let mut v = vec![0u8; total_query_strings * descriptor_size_bytes];
    file.read_exact(&mut v).expect(
        "unable to read enough search descriptors from the file; add more descriptors to file",
    );
    let query_strings: Rc<Vec<Hamming<Bits256>>> = Rc::new(
        v.chunks_exact(descriptor_size_bytes)
            .map(|b| {
                let mut arr = [0; 32];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                Hamming(Bits256(arr))
            })
            .collect(),
    );
    eprintln!("Done.");

    eprintln!("Generating HNSWs...");
    let hnsw_map = Rc::new(HashMap::<_, _>::from_iter(all_sizes.clone().map(|total| {
        eprintln!("Generating HNSW size {}...", total);
        let range = 0..total;
        let mut hnsw: HNSW<Hamming<Bits256>> = HNSW::new();
        let mut searcher = Searcher::default();
        for i in range.clone() {
            hnsw.insert(search_space[i], &mut searcher);
        }
        (total, hnsw)
    })));
    eprintln!("Done.");
    c.bench(
        "neighbors",
        ParameterizedBenchmark::new(
            "2_nn_linear_handrolled",
            {
                let query_strings = query_strings.clone();
                let search_space = search_space.clone();
                move |bencher: &mut Bencher, &total: &usize| {
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    bencher.iter(|| {
                        let search_feature = cycle_range.next().unwrap();
                        let mut best_distance = !0;
                        let mut next_best_distance = !0;
                        let mut best = 0;
                        let mut next_best = 0;
                        for (ix, feature) in search_space[0..total].iter().enumerate() {
                            let distance = search_feature.distance(feature);
                            if distance < next_best_distance {
                                if distance < best_distance {
                                    next_best_distance = best_distance;
                                    next_best = best;
                                    best_distance = distance;
                                    best = ix;
                                } else {
                                    next_best_distance = distance;
                                    next_best = ix;
                                }
                            }
                        }
                        (best, next_best, best_distance, next_best_distance)
                    });
                }
            },
            all_sizes,
        )
        .with_function("2_nn_DiscreteHNSW", {
            {
                let hnsw_map = hnsw_map.clone();
                let query_strings = query_strings.clone();
                move |bencher: &mut Bencher, total: &usize| {
                    let hnsw = &hnsw_map[total];
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut searcher = Searcher::default();
                    bencher.iter(|| {
                        let feature = cycle_range.next().unwrap();
                        let mut neighbors = [Neighbor::invalid(); 2];
                        hnsw.nearest(&feature, 24, &mut searcher, &mut neighbors)
                            .len()
                    });
                }
            }
        })
        .with_function("2_nn_linear_FixedHammingHeap", {
            {
                let query_strings = query_strings.clone();
                let search_space = search_space.clone();
                move |bencher: &mut Bencher, &total: &usize| {
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut nearest = FixedHammingHeap::new_distances(257);
                    bencher.iter(|| {
                        nearest.set_capacity(2);
                        nearest.clear();
                        let search_feature = cycle_range.next().unwrap();
                        for (ix, feature) in search_space[0..total].iter().enumerate() {
                            nearest.push(search_feature.distance(feature), ix as u32);
                        }
                        nearest.len()
                    });
                }
            }
        })
        .with_function("2_nn_linear_FixedDiscreteCandidates", {
            {
                let query_strings = query_strings.clone();
                let search_space = search_space.clone();
                move |bencher: &mut Bencher, &total: &usize| {
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut nearest = CandidatesVec::default();
                    bencher.iter(|| {
                        nearest.set_cap(2);
                        nearest.clear();
                        let search_feature = cycle_range.next().unwrap();
                        for (index, feature) in search_space[0..total].iter().enumerate() {
                            let distance = search_feature.distance(feature);
                            nearest.push(Neighbor { index, distance });
                        }
                        nearest.len()
                    });
                }
            }
        })
        .with_function("10_nn_DiscreteHNSW", {
            {
                let hnsw_map = hnsw_map;
                let query_strings = query_strings.clone();
                move |bencher: &mut Bencher, total: &usize| {
                    let hnsw = &hnsw_map[total];
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut searcher = Searcher::default();
                    bencher.iter(|| {
                        let feature = cycle_range.next().unwrap();
                        let mut neighbors = [Neighbor::invalid(); 10];
                        hnsw.nearest(&feature, 24, &mut searcher, &mut neighbors)
                            .len()
                    });
                }
            }
        })
        .with_function("10_nn_linear_FixedHammingHeap", {
            {
                let query_strings = query_strings.clone();
                let search_space = search_space.clone();
                move |bencher: &mut Bencher, &total: &usize| {
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut nearest = FixedHammingHeap::new_distances(257);
                    bencher.iter(|| {
                        nearest.set_capacity(10);
                        nearest.clear();
                        let search_feature = cycle_range.next().unwrap();
                        for (ix, feature) in search_space[0..total].iter().enumerate() {
                            nearest.push(search_feature.distance(feature), ix as u32);
                        }
                        nearest.len()
                    });
                }
            }
        })
        .with_function("10_nn_linear_FixedDiscreteCandidates", {
            {
                let query_strings = query_strings;
                let search_space = search_space;
                move |bencher: &mut Bencher, &total: &usize| {
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut nearest = CandidatesVec::default();
                    bencher.iter(|| {
                        nearest.set_cap(10);
                        nearest.clear();
                        let search_feature = cycle_range.next().unwrap();
                        for (index, feature) in search_space[0..total].iter().enumerate() {
                            let distance = search_feature.distance(feature);
                            nearest.push(Neighbor { index, distance });
                        }
                        nearest.len()
                    });
                }
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
