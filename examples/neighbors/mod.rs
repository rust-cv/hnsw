use byteorder::{ByteOrder, NativeEndian};
use criterion::*;
use hamming_heap::FixedHammingHeap;
use hnsw::*;
use packed_simd::u128x2;
use std::collections::HashMap;
use std::io::Read;
use std::iter::FromIterator;
use std::rc::Rc;

fn make_u128x2(bytes: &[u8]) -> Hamming<u128x2> {
    Hamming(
        [
            NativeEndian::read_u128(&bytes[0..16]),
            NativeEndian::read_u128(&bytes[16..32]),
        ]
        .into(),
    )
}

fn bench_neighbors(c: &mut Criterion) {
    let space_mags = 0..=19;
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
    let search_space: Vec<Hamming<u128x2>> = v
        .chunks_exact(descriptor_size_bytes)
        .map(make_u128x2)
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
    let query_strings: Rc<Vec<Hamming<u128x2>>> = Rc::new(
        v.chunks_exact(descriptor_size_bytes)
            .map(make_u128x2)
            .collect(),
    );
    eprintln!("Done.");

    eprintln!("Generating HNSWs...");
    let hnsw_map = Rc::new(HashMap::<_, _>::from_iter(all_sizes.clone().map(|total| {
        eprintln!("Generating HNSW size {}...", total);
        let range = 0..total;
        let mut hnsw: DiscreteHNSW<Hamming<u128x2>> = DiscreteHNSW::new();
        let mut searcher = DiscreteSearcher::default();
        for i in range.clone() {
            hnsw.insert(search_space[i], &mut searcher);
        }
        (total, hnsw)
    })));
    eprintln!("Done.");
    c.bench(
        "neighbors",
        ParameterizedBenchmark::new(
            "2_nn_hnsw",
            {
                let hnsw_map = hnsw_map.clone();
                let query_strings = query_strings.clone();
                move |bencher: &mut Bencher, total: &usize| {
                    let hnsw = &hnsw_map[total];
                    let mut cycle_range = query_strings.iter().cloned().cycle();
                    let mut searcher = DiscreteSearcher::default();
                    bencher.iter(|| {
                        let feature = cycle_range.next().unwrap();
                        let mut neighbors = [0; 2];
                        hnsw.nearest(&feature, 24, &mut searcher, &mut neighbors)
                            .len()
                    });
                }
            },
            all_sizes,
        )
        .with_function("2_nn_linear", {
            move |bencher: &mut Bencher, &total: &usize| {
                let mut cycle_range = query_strings.iter().cloned().cycle();
                let mut nearest = FixedHammingHeap::<
                    <Hamming<u128x2> as DiscreteDistance>::Distances,
                    u32,
                >::default();
                bencher.iter(|| {
                    nearest.set_capacity(2);
                    nearest.clear();
                    let search_feature = cycle_range.next().unwrap();
                    for (ix, feature) in search_space[0..total].iter().enumerate() {
                        nearest.push(
                            DiscreteDistance::discrete_distance(&search_feature, feature),
                            ix as u32,
                        );
                    }
                    nearest.len()
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
