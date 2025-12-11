use hnsw::{FeatureStore, Hnsw, Params, Searcher};
use rand::Rng;
use rand_pcg::Pcg64;
use space::Metric;
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use structopt::StructOpt;

struct Euclidean;

impl Metric<[f32; 128]> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &[f32; 128], b: &[f32; 128]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits()
    }
}

const FEATURE_SIZE: usize = 128;
const FEATURE_BYTES: usize = FEATURE_SIZE * std::mem::size_of::<f32>();

/// A disk-based feature store that reads features from disk on every access.
/// This uses NO RAM for feature storage - only a small buffer for reading.
struct DiskFeatureStore {
    file: File,
    len: usize,
}

impl DiskFeatureStore {
    fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        Ok(Self { file, len: 0 })
    }
}

impl FeatureStore<[f32; 128]> for DiskFeatureStore {
    fn get(&self, index: usize) -> &[f32; 128] {
        thread_local! {
            static BUFFER: std::cell::UnsafeCell<[f32; 128]> = std::cell::UnsafeCell::new([0.0; 128]);
        }

        BUFFER.with(|buf| {
            let buffer = unsafe { &mut *buf.get() };

            let mut file = &self.file;
            let offset = index * FEATURE_BYTES;
            file.seek(SeekFrom::Start(offset as u64)).unwrap();

            let mut bytes = [0u8; FEATURE_BYTES];
            file.read_exact(&mut bytes).unwrap();

            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buffer[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }

            unsafe { &*(buffer as *const [f32; 128]) }
        })
    }

    fn push(&mut self, feature: [f32; 128]) {
        self.file.seek(SeekFrom::End(0)).unwrap();
        let bytes: Vec<u8> = feature.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.file.write_all(&bytes).unwrap();
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
    name = "lazy_memory_comparison",
    about = "Measures memory usage of HNSW with different storage backends"
)]
struct Opt {
    /// Number of vectors to insert.
    #[structopt(short = "n", long = "num-vectors", default_value = "100000")]
    num_vectors: usize,
    /// Use disk-based feature storage instead of in-memory Vec<T>.
    #[structopt(long = "disk")]
    disk: bool,
}

fn generate_random_vectors(count: usize) -> Vec<[f32; 128]> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            let mut v = [0.0f32; 128];
            for x in v.iter_mut() {
                *x = rng.gen();
            }
            v
        })
        .collect()
}

fn get_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if parts.len() >= 2 {
                let rss_pages: usize = parts[1].parse().unwrap_or(0);
                return rss_pages * 4096;
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(rss_str) = String::from_utf8(output.stdout) {
                if let Ok(rss_kb) = rss_str.trim().parse::<usize>() {
                    return rss_kb * 1024;
                }
            }
        }
    }

    0
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn run_with_vec(num_vectors: usize) {
    let params = Params::new().ef_construction(16);

    println!("=== Vec<T> Storage ===");
    println!("Generating {} random vectors...", num_vectors);
    let vectors = generate_random_vectors(num_vectors);

    let mem_before = get_memory_usage();

    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut hnsw: Hnsw<Euclidean, [f32; 128], Pcg64, 12, 24> =
        Hnsw::new_params_and_prng(Euclidean, params, prng);
    let mut searcher = Searcher::default();

    println!("Building HNSW...");
    for v in &vectors {
        hnsw.insert(*v, &mut searcher);
    }

    let mem_after = get_memory_usage();
    let mem_delta = mem_after.saturating_sub(mem_before);

    println!();
    println!("Vectors:          {}", num_vectors);
    println!("Feature size:     {} bytes", FEATURE_BYTES);
    println!("Memory used:      {}", format_bytes(mem_delta));
    println!("Expected features: {}", format_bytes(num_vectors * FEATURE_BYTES));
}

fn run_with_disk(num_vectors: usize) -> std::io::Result<()> {
    let params = Params::new().ef_construction(16);
    let disk_path = "/tmp/hnsw_disk_features.bin";

    println!("=== Disk-based Storage ===");
    println!("Generating {} random vectors...", num_vectors);
    let vectors = generate_random_vectors(num_vectors);

    let mem_before = get_memory_usage();

    let storage = DiskFeatureStore::new(disk_path)?;
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let mut hnsw: Hnsw<Euclidean, [f32; 128], Pcg64, 12, 24, DiskFeatureStore> =
        Hnsw::new_with_storage_and_params(Euclidean, storage, params, prng);
    let mut searcher = Searcher::default();

    println!("Building HNSW...");
    for v in &vectors {
        hnsw.insert(*v, &mut searcher);
    }

    let mem_after = get_memory_usage();
    let mem_delta = mem_after.saturating_sub(mem_before);

    println!();
    println!("Vectors:          {}", num_vectors);
    println!("Feature size:     {} bytes", FEATURE_BYTES);
    println!("Memory used:      {}", format_bytes(mem_delta));
    println!("Expected features: {}", format_bytes(num_vectors * FEATURE_BYTES));

    std::fs::remove_file(disk_path).ok();
    Ok(())
}

fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();

    if opt.disk {
        run_with_disk(opt.num_vectors)?;
    } else {
        run_with_vec(opt.num_vectors);
    }

    Ok(())
}
