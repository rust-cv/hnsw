use byteorder::{ByteOrder, NativeEndian};
use generic_array::{typenum, ArrayLength};
use gnuplot::*;
use hnsw::*;
use itertools::Itertools;
use packed_simd::{u128x2, u128x4};
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::cell::RefCell;
use std::io::Read;
use std::path::PathBuf;
use structopt::StructOpt;

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
}

fn process<T: DiscreteDistance + Clone, M: ArrayLength<u32>, M0: ArrayLength<u32>>(
	opt: &Opt,
	conv: fn(&[u8]) -> T,
) -> (Vec<f64>, Vec<f64>) {
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
			.map(|n: u8| n)
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
			.map(|n: u8| n)
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
	let correct_worst_distances: Vec<u32> = query_strings
		.iter()
		.cloned()
		.map(|feature| {
			let mut v = vec![];
			for distance in search_space
				.iter()
				.map(|n| T::discrete_distance(n, &feature))
			{
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
	let mut hnsw: DiscreteHNSW<T, M, M0> = DiscreteHNSW::new();
	let mut searcher: DiscreteSearcher<T> = DiscreteSearcher::default();
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
			let dest = vec![!0; opt.k];
			let stats = easybench::bench_env(dest, |mut dest| {
				let mut refmut = state.borrow_mut();
				let (searcher, query) = &mut *refmut;
				let (ix, query_feature) = query.next().unwrap();
				let correct_worst_distance = correct_worst_distances[ix];
				// Go through all the features.
				for &mut feature_ix in hnsw.nearest(&query_feature, ef, searcher, &mut dest) {
					// Any feature that is less than or equal to the worst real nearest neighbor distance is correct.
					if T::discrete_distance(&search_space[feature_ix as usize], &query_feature)
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
	( $opt:expr, $m:ty, $m0:ty ) => {
		match $opt.bitstring_length {
			8 => process::<_, $m, $m0>(&$opt, |b| Hamming(b[0])),
			16 => process::<_, $m, $m0>(&$opt, |b| Hamming(NativeEndian::read_u16(b))),
			32 => process::<_, $m, $m0>(&$opt, |b| Hamming(NativeEndian::read_u32(b))),
			64 => process::<_, $m, $m0>(&$opt, |b| Hamming(NativeEndian::read_u64(b))),
			128 => process::<_, $m, $m0>(&$opt, |b| Hamming(NativeEndian::read_u128(b))),
			256 => process::<_, $m, $m0>(&$opt, make_u128x2),
			512 => process::<_, $m, $m0>(&$opt, make_u128x4),
			_ => panic!("error: incorrect bitstring_length, see --help for choices"),
			}
	};
}

fn main() {
	let opt = Opt::from_args();

	fn make_u128x2(bytes: &[u8]) -> Hamming<u128x2> {
		Hamming(
			[
				byteorder::NativeEndian::read_u128(&bytes[0..16]),
				byteorder::NativeEndian::read_u128(&bytes[16..32]),
			]
			.into(),
		)
	}

	fn make_u128x4(bytes: &[u8]) -> Hamming<u128x4> {
		Hamming(
			[
				byteorder::NativeEndian::read_u128(&bytes[0..16]),
				byteorder::NativeEndian::read_u128(&bytes[16..32]),
				byteorder::NativeEndian::read_u128(&bytes[32..48]),
				byteorder::NativeEndian::read_u128(&bytes[48..64]),
			]
			.into(),
		)
	}

	let (recalls, times) = {
		use typenum::*;
		// This can be increased indefinitely at the expense of compile time.
		match opt.m {
			4 => process_m!(opt, U4, U8),
			8 => process_m!(opt, U8, U16),
			12 => process_m!(opt, U12, U24),
			16 => process_m!(opt, U16, U32),
			20 => process_m!(opt, U20, U40),
			24 => process_m!(opt, U24, U48),
			28 => process_m!(opt, U28, U56),
			32 => process_m!(opt, U32, U64),
			36 => process_m!(opt, U36, U72),
			40 => process_m!(opt, U40, U80),
			44 => process_m!(opt, U44, U88),
			48 => process_m!(opt, U48, U96),
			52 => process_m!(opt, U52, U104),
			_ => {
				eprintln!("Only M between 4 and 52 inclusive and multiples of 4 are allowed");
				return;
			}
		}
	};

	let mut fg = Figure::new();

	fg.axes2d()
		.set_title(
			&format!(
				"{}-NN Recall Graph (bits = {}, size = {}, M = {})",
				opt.k, opt.bitstring_length, opt.size, opt.m
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

	fg.show();
}
