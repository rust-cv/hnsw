use bitarray::BitArray;
use criterion::*;
use hnsw::*;
use rand_pcg::Pcg64;
use space::*;
use std::io::Read;

fn bench_neighbors(c: &mut Criterion) {
    // Set up knn benchmark group.
    let mut group = c.benchmark_group("knn");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let space_mags = 0..=16;
    let all_sizes = (space_mags).map(|n| 2usize.pow(n));
    let max_linear_size = 2usize.pow(14);
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
    let search_space: Vec<BitArray<32>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(|b| {
            let mut arr = [0; 32];
            for (d, &s) in arr.iter_mut().zip(b) {
                *d = s;
            }
            BitArray::new(arr)
        })
        .collect();
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
    let query_strings: Vec<BitArray<32>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(|b| {
            let mut arr = [0; 32];
            for (d, &s) in arr.iter_mut().zip(b) {
                *d = s;
            }
            BitArray::new(arr)
        })
        .collect();
    eprintln!("Done.");

    // Run each test on each size of search set.
    for size in all_sizes {
        {
            // This is a workaround for the fact that criterion wont generate a combined plot without everything being present.
            let parameter = size;
            let size = if size <= max_linear_size { size } else { 1 };

            // Run 2_nn_linear_handrolled test.
            let mut cycle_range = query_strings.iter().cloned().cycle();
            group.bench_with_input(
                BenchmarkId::new("2_nn_linear_handrolled", parameter),
                &(),
                |b, _| {
                    b.iter(|| {
                        let search_feature = cycle_range.next().unwrap();
                        let mut best_distance = !0;
                        let mut next_best_distance = !0;
                        let mut best = 0;
                        let mut next_best = 0;
                        for (ix, feature) in search_space[0..size].iter().enumerate() {
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
                    })
                },
            );

            // Run 2_nn_linear_Vec test.
            let mut cycle_range = query_strings.iter().cloned().cycle();
            group.bench_with_input(
                BenchmarkId::new("2_nn_LinearKnn", parameter),
                &(),
                |b, _| {
                    b.iter(|| {
                        let search_feature = cycle_range.next().unwrap();
                        LinearKnn(search_space[..size].iter()).knn(&search_feature, 2)
                    })
                },
            );

            // Run 10_nn_linear_Vec test.
            let mut cycle_range = query_strings.iter().cloned().cycle();
            group.bench_with_input(
                BenchmarkId::new("10_nn_LinearKnn", parameter),
                &(),
                |b, _| {
                    b.iter(|| {
                        let search_feature = cycle_range.next().unwrap();
                        LinearKnn(search_space[..size].iter()).knn(&search_feature, 10)
                    })
                },
            );
        }

        eprintln!("Generating HNSW size {}...", size);
        let mut hnsw: Hnsw<BitArray<32>, Pcg64, 12, 24> = Hnsw::new();
        let mut searcher = Searcher::default();
        for &item in &search_space[0..size] {
            hnsw.insert(item, &mut searcher);
        }
        group.throughput(Throughput::Elements(size as u64));

        // Run 2_nn_DiscreteHNSW test.
        let mut cycle_range = query_strings.iter().cloned().cycle();
        let mut searcher = Searcher::default();
        group.bench_with_input(BenchmarkId::new("2_nn_DiscreteHNSW", size), &(), |b, _| {
            b.iter(|| {
                let feature = cycle_range.next().unwrap();
                let mut neighbors = [Neighbor {
                    index: !0,
                    distance: !0,
                }; 2];
                hnsw.nearest(&feature, 24, &mut searcher, &mut neighbors)
                    .len()
            })
        });

        // Run 10_nn_DiscreteHNSW test.
        let mut cycle_range = query_strings.iter().cloned().cycle();
        let mut searcher = Searcher::default();
        group.bench_with_input(BenchmarkId::new("10_nn_DiscreteHNSW", size), &(), |b, _| {
            b.iter(|| {
                let feature = cycle_range.next().unwrap();
                let mut neighbors = [Neighbor {
                    index: !0,
                    distance: !0,
                }; 10];
                hnsw.nearest(&feature, 24, &mut searcher, &mut neighbors)
                    .len()
            })
        });
    }
}

fn config() -> Criterion {
    Criterion::default().sample_size(32)
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_neighbors
}
