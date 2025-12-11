use bitarray::{BitArray, Hamming};
use gnuplot::*;
use hnsw::*;
use itertools::Itertools;
use memmap2::MmapMut;
use num_traits::Zero;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::*;
use std::cell::RefCell;
use std::fs::OpenOptions;
use std::io::Read;
use std::path::PathBuf;
use structopt::StructOpt;


struct MmapBitArrayStore<T> {
    mmap: MmapMut,
    len: usize,
    capacity: usize,
    stride: usize,
    _marker: std::marker::PhantomData<T>,
}

const ALIGNMENT: usize = 64;

// this is very suboptimal but works for testing purposes
fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl<T> MmapBitArrayStore<T> {
    fn new(path: &str, capacity: usize, item_size: usize) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Align each item to 64 bytes for SIMD
        let stride = align_up(item_size, ALIGNMENT);
        // Add padding at start to ensure first item is aligned
        let total_size = ALIGNMENT + capacity * stride;
        file.set_len(total_size as u64)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            mmap,
            len: 0,
            capacity,
            stride,
            _marker: std::marker::PhantomData,
        })
    }

    fn aligned_offset(&self, index: usize) -> usize {
        // Find offset that gives 64-byte alignment
        let base = self.mmap.as_ptr() as usize;
        let aligned_base = align_up(base, ALIGNMENT);
        let padding = aligned_base - base;
        padding + index * self.stride
    }
}

impl<const N: usize> FeatureStore<BitArray<N>> for MmapBitArrayStore<BitArray<N>> {
    fn get(&self, index: usize) -> &BitArray<N> {
        let offset = self.aligned_offset(index);
        unsafe { &*(self.mmap[offset..].as_ptr() as *const BitArray<N>) }
    }

    fn push(&mut self, feature: BitArray<N>) {
        // we dont need to be sophisticated for this dummy example
        assert!(self.len < self.capacity, "MmapBitArrayStore capacity exceeded");
        let offset = self.aligned_offset(self.len);
        self.mmap[offset..offset + N].copy_from_slice(feature.bytes());
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "recall_discrete",
    about = "Generates recall graphs for DiscreteHNSW"
)]
struct Opt {
    /// The value of M to use.
    ///
    /// This can only be between 4 and 52 inclusive and a multiple of 4.
    /// M0 is set to 2 * M.
    #[structopt(short = "m", long = "max_edges", default_value = "24")]
    m: usize,
    /// The dataset size to test on.
    #[structopt(short = "s", long = "size", default_value = "10000")]
    size: usize,
    /// Total number of query bitstrings.
    ///
    /// The higher this is, the better the quality of the output data and statistics.
    #[structopt(short = "q", long = "queries", default_value = "10000")]
    num_queries: usize,
    /// The bitstring length.
    ///
    /// This is the length of bitstrings in bits. The descriptor_stride (-d)
    /// parameter must exceed this value.
    ///
    /// Possible values:
    /// - 8
    /// - 16
    /// - 32
    /// - 64
    /// - 128
    /// - 256
    /// - 512
    #[structopt(short = "l", long = "bitstring_length", default_value = "256")]
    bitstring_length: usize,
    /// The beginning ef value.
    #[structopt(short = "b", long = "beginning_ef", default_value = "1")]
    beginning_ef: usize,
    /// The ending ef value.
    #[structopt(short = "e", long = "ending_ef", default_value = "64")]
    ending_ef: usize,
    /// The number of nearest neighbors.
    #[structopt(short = "k", long = "neighbors", default_value = "2")]
    k: usize,
    /// Use the following file to load the search space.
    #[structopt(short = "f", long = "file")]
    file: Option<PathBuf>,
    /// The descriptor stride length in bytes.
    ///
    /// AKAZE: 61
    /// ORB: 32
    ///
    #[structopt(short = "d", long = "descriptor_stride", default_value = "61")]
    descriptor_stride: usize,
    /// efConstruction controlls the quality of the graph at build-time.
    #[structopt(short = "c", long = "ef_construction", default_value = "400")]
    ef_construction: usize,
    /// Use disk-based feature storage instead of in-memory Vec<T>.
    /// This tests the FeatureStore trait with a disk backend.
    /// Note: Does not support file input (-f).
    #[structopt(long = "disk")]
    disk: bool,
}

fn process<T: Clone, S: FeatureStore<T>, const M: usize, const M0: usize>(
    opt: &Opt,
    conv: fn(&[u8]) -> T,
    storage: S,
) -> (Vec<f64>, Vec<f64>)
where
    Hamming: Metric<T>,
{
    assert!(
        opt.k <= opt.size,
        "You must choose a dataset size larger or equal to the test search size"
    );
    let rng = Pcg64::from_seed([5; 32]);

    let (search_space, query_strings): (Vec<T>, Vec<T>) = if let Some(filepath) = &opt.file {
        eprintln!(
            "Reading {} search space descriptors of size {} bytes from file \"{}\"...",
            opt.size,
            opt.descriptor_stride,
            filepath.display()
        );
        let mut file = std::fs::File::open(filepath).expect("unable to open file");
        let mut v = vec![0u8; opt.size * opt.descriptor_stride];
        file.read_exact(&mut v).expect(
            "unable to read enough search descriptors from the file (try decreasing -s/-q)",
        );
        let search_space = v.chunks_exact(opt.descriptor_stride).map(conv).collect();
        eprintln!("Done.");

        eprintln!(
            "Reading {} query descriptors of size {} bytes from file \"{}\"...",
            opt.num_queries,
            opt.descriptor_stride,
            filepath.display()
        );
        let mut v = vec![0u8; opt.num_queries * opt.descriptor_stride];
        file.read_exact(&mut v)
            .expect("unable to read enough query descriptors from the file (try decreasing -q/-s)");
        let query_strings = v.chunks_exact(opt.descriptor_stride).map(conv).collect();
        eprintln!("Done.");

        (search_space, query_strings)
    } else {
        eprintln!("Generating {} random bitstrings...", opt.size);
        let search_space: Vec<T> = rng
            .sample_iter(&Standard)
            .chunks(64)
            .into_iter()
            .map(|bs| conv(&bs.collect::<Vec<u8>>()))
            .take(opt.size)
            .collect();
        eprintln!("Done.");

        // Create another RNG to prevent potential correlation.
        let rng = Pcg64::from_seed([6; 32]);

        eprintln!(
            "Generating {} independent random query strings...",
            opt.num_queries
        );
        let query_strings: Vec<T> = rng
            .sample_iter(&Standard)
            .chunks(64)
            .into_iter()
            .map(|bs| conv(&bs.collect::<Vec<u8>>()))
            .take(opt.size)
            .collect();
        eprintln!("Done.");
        (search_space, query_strings)
    };

    eprintln!(
        "Computing the correct nearest neighbor distance for all {} queries...",
        opt.num_queries
    );
    let correct_worst_distances: Vec<_> = query_strings
        .iter()
        .cloned()
        .map(|feature| {
            let mut v = vec![];
            for distance in search_space.iter().map(|n| Hamming.distance(n, &feature)) {
                let pos = v.binary_search(&distance).unwrap_or_else(|e| e);
                v.insert(pos, distance);
                if v.len() > opt.k {
                    v.resize_with(opt.k, || unreachable!());
                }
            }
            // Get the worst distance
            v.into_iter().take(opt.k).last().unwrap()
        })
        .collect();
    eprintln!("Done.");

    eprintln!("Generating HNSW...");
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut hnsw: Hnsw<Hamming, T, Pcg64, M, M0, S> =
        Hnsw::new_with_storage_and_params(Hamming, storage, Params::new().ef_construction(opt.ef_construction), prng);
    let mut searcher: Searcher<_> = Searcher::default();
    for feature in &search_space {
        hnsw.insert(feature.clone(), &mut searcher);
    }
    eprintln!("Done.");

    eprintln!("Computing recall graph...");
    let efs = opt.beginning_ef..=opt.ending_ef;
    let state = RefCell::new((searcher, query_strings.iter().cloned().enumerate().cycle()));
    let (recalls, times): (Vec<f64>, Vec<f64>) = efs
        .map(|ef| {
            let correct = RefCell::new(0usize);
            let dest = vec![
                Neighbor {
                    index: !0usize,
                    distance: <Hamming as Metric<T>>::Unit::zero(),
                };
                opt.k
            ];
            let stats = easybench::bench_env(dest, |dest| {
                let mut refmut = state.borrow_mut();
                let (searcher, query) = &mut *refmut;
                let (ix, query_feature) = query.next().unwrap();
                let correct_worst_distance = correct_worst_distances[ix];
                // Go through all the features.
                for &mut neighbor in hnsw.nearest(&query_feature, ef, searcher, dest) {
                    // Any feature that is less than or equal to the worst real nearest neighbor distance is correct.
                    if Hamming.distance(&search_space[neighbor.index], &query_feature)
                        <= correct_worst_distance
                    {
                        *correct.borrow_mut() += 1;
                    }
                }
            });
            (stats, correct.into_inner())
        })
        .fold(
            (vec![], vec![]),
            |(mut recalls, mut times), (stats, correct)| {
                times.push((stats.ns_per_iter * 0.1f64.powi(9)).recip());
                // The maximum number of correct nearest neighbors is
                recalls.push(correct as f64 / (stats.iterations * opt.k) as f64);
                (recalls, times)
            },
        );
    eprintln!("Done.");

    (recalls, times)
}

macro_rules! process_m {
    ( $opt:expr, $m:expr, $m0:expr ) => {
        match $opt.bitstring_length {
            128 => process::<BitArray<16>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 16];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, Vec::new()),
            256 => process::<BitArray<32>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 32];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, Vec::new()),
            512 => process::<BitArray<64>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 64];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, Vec::new()),
            _ => panic!("error: incorrect bitstring_length, see --help for choices"),
        }
    };
}

macro_rules! process_mmap_m {
    ( $opt:expr, $m:expr, $m0:expr ) => {
        match $opt.bitstring_length {
            128 => process::<BitArray<16>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 16];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, MmapBitArrayStore::new("/tmp/recall_discrete_mmap.bin", $opt.size, 16).unwrap()),
            256 => process::<BitArray<32>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 32];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, MmapBitArrayStore::new("/tmp/recall_discrete_mmap.bin", $opt.size, 32).unwrap()),
            512 => process::<BitArray<64>, _, $m, $m0>(&$opt, |b| {
                let mut arr = [0; 64];
                for (d, &s) in arr.iter_mut().zip(b) {
                    *d = s;
                }
                BitArray::new(arr)
            }, MmapBitArrayStore::new("/tmp/recall_discrete_mmap.bin", $opt.size, 64).unwrap()),
            _ => panic!("error: incorrect bitstring_length, see --help for choices"),
        }
    };
}

fn main() {
    let opt = Opt::from_args();

    let (recalls, times, storage_type) = if opt.disk {
        let (r, t) = match opt.m {
            4 => process_mmap_m!(opt, 4, 8),
            8 => process_mmap_m!(opt, 8, 16),
            12 => process_mmap_m!(opt, 12, 24),
            16 => process_mmap_m!(opt, 16, 32),
            20 => process_mmap_m!(opt, 20, 40),
            24 => process_mmap_m!(opt, 24, 48),
            28 => process_mmap_m!(opt, 28, 56),
            32 => process_mmap_m!(opt, 32, 64),
            36 => process_mmap_m!(opt, 36, 72),
            40 => process_mmap_m!(opt, 40, 80),
            44 => process_mmap_m!(opt, 44, 88),
            48 => process_mmap_m!(opt, 48, 96),
            52 => process_mmap_m!(opt, 52, 104),
            _ => {
                eprintln!("Only M between 4 and 52 inclusive and multiples of 4 are allowed");
                return;
            }
        };
        (r, t, "MmapBitArrayStore")
    } else {
        let (r, t) = match opt.m {
            4 => process_m!(opt, 4, 8),
            8 => process_m!(opt, 8, 16),
            12 => process_m!(opt, 12, 24),
            16 => process_m!(opt, 16, 32),
            20 => process_m!(opt, 20, 40),
            24 => process_m!(opt, 24, 48),
            28 => process_m!(opt, 28, 56),
            32 => process_m!(opt, 32, 64),
            36 => process_m!(opt, 36, 72),
            40 => process_m!(opt, 40, 80),
            44 => process_m!(opt, 44, 88),
            48 => process_m!(opt, 48, 96),
            52 => process_m!(opt, 52, 104),
            _ => {
                eprintln!("Only M between 4 and 52 inclusive and multiples of 4 are allowed");
                return;
            }
        };
        (r, t, "Vec<T>")
    };

    let mut fg = Figure::new();

    fg.axes2d()
        .set_title(
            &format!(
                "{}-NN Recall Graph (bits = {}, size = {}, M = {}, storage = {})",
                opt.k, opt.bitstring_length, opt.size, opt.m, storage_type
            ),
            &[],
        )
        .set_x_label("Recall Rate", &[])
        .set_y_label("Lookups per second", &[])
        .lines(&recalls, &times, &[LineWidth(2.0), Color("blue")])
        .set_y_ticks(Some((Auto, 2)), &[], &[])
        .set_grid_options(true, &[LineStyle(DotDotDash), Color("black")])
        .set_minor_grid_options(&[LineStyle(SmallDot), Color("red")])
        .set_x_grid(true)
        .set_y_grid(true)
        .set_y_minor_grid(true);

    fg.show().ok(); // Don't panic if gnuplot unavailable
}
