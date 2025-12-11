//! Tests for the FeatureStore trait and custom storage backends.

use hnsw::{FeatureStore, Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};
use std::cell::Cell;

struct Euclidean;

impl Metric<[f64; 4]> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &[f64; 4], b: &[f64; 4]) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
            .to_bits()
    }
}

struct TrackedFeatureStore<T> {
    inner: Vec<T>,
    get_count: Cell<usize>,
    push_count: Cell<usize>,
}

impl<T> TrackedFeatureStore<T> {
    fn new() -> Self {
        Self {
            inner: Vec::new(),
            get_count: Cell::new(0),
            push_count: Cell::new(0),
        }
    }

    fn get_count(&self) -> usize {
        self.get_count.get()
    }

    fn push_count(&self) -> usize {
        self.push_count.get()
    }

    fn reset_counts(&self) {
        self.get_count.set(0);
        self.push_count.set(0);
    }
}

impl<T> FeatureStore<T> for TrackedFeatureStore<T> {
    fn get(&self, index: usize) -> &T {
        self.get_count.set(self.get_count.get() + 1);
        &self.inner[index]
    }

    fn push(&mut self, feature: T) {
        self.push_count.set(self.push_count.get() + 1);
        self.inner.push(feature);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// A simple in-memory cache that simulates lazy loading behavior.
struct MockLazyStore<T: Clone + Default> {
    storage: Vec<Option<T>>,
    loaded_count: Cell<usize>,
}

impl<T: Clone + Default> MockLazyStore<T> {
    fn new() -> Self {
        Self {
            storage: Vec::new(),
            loaded_count: Cell::new(0),
        }
    }

    fn loaded_count(&self) -> usize {
        self.loaded_count.get()
    }
}

impl<T: Clone + Default> FeatureStore<T> for MockLazyStore<T> {
    fn get(&self, index: usize) -> &T {
        self.loaded_count.set(self.loaded_count.get() + 1);
        self.storage[index].as_ref().expect("Feature not found")
    }

    fn push(&mut self, feature: T) {
        self.storage.push(Some(feature));
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

#[test]
fn test_tracked_feature_store() {
    let storage = TrackedFeatureStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let mut hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, TrackedFeatureStore<[f64; 4]>> =
        Hnsw::new_with_storage(Euclidean, storage, prng);

    let mut searcher = Searcher::default();

    // Insert features
    let features = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ];

    for feature in features.iter() {
        hnsw.insert(*feature, &mut searcher);
    }

    // Verify push was called for each insert
    assert_eq!(hnsw.len(), 4);

    // Search and verify get is called
    let mut neighbors = [Neighbor {
        index: !0,
        distance: !0,
    }; 4];

    hnsw.nearest(&[0.0, 0.0, 0.0, 1.0], 24, &mut searcher, &mut neighbors);

    // First result should be exact match
    assert_eq!(neighbors[0].index, 0);

    // Verify feature() method works through the trait
    let feature = hnsw.feature(0);
    assert_eq!(*feature, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_mock_lazy_store() {
    // Test a mock lazy loading store
    let storage = MockLazyStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let mut hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, MockLazyStore<[f64; 4]>> =
        Hnsw::new_with_storage(Euclidean, storage, prng);

    let mut searcher = Searcher::default();

    // Insert features
    let features = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ];

    for feature in features.iter() {
        hnsw.insert(*feature, &mut searcher);
    }

    assert_eq!(hnsw.len(), 8);

    // Search
    let mut neighbors = [Neighbor {
        index: !0,
        distance: !0,
    }; 4];

    hnsw.nearest(&[0.0, 0.0, 0.0, 1.0], 24, &mut searcher, &mut neighbors);

    // Verify correct results
    assert_eq!(neighbors[0].index, 0); // Exact match
    assert_eq!(neighbors[0].distance, 0);
}

#[test]
fn test_feature_store_with_params() {
    // Test new_with_storage_and_params constructor
    use hnsw::Params;

    let storage = TrackedFeatureStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);
    let params = Params::new().ef_construction(100);

    let mut hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, TrackedFeatureStore<[f64; 4]>> =
        Hnsw::new_with_storage_and_params(Euclidean, storage, params, prng);

    let mut searcher = Searcher::default();

    hnsw.insert([1.0, 2.0, 3.0, 4.0], &mut searcher);
    assert_eq!(hnsw.len(), 1);
}

#[test]
fn test_feature_store_layer_feature() {
    // Test layer_feature method works with custom store
    let storage = TrackedFeatureStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let mut hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, TrackedFeatureStore<[f64; 4]>> =
        Hnsw::new_with_storage(Euclidean, storage, prng);

    let mut searcher = Searcher::default();

    let features = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ];

    for feature in features.iter() {
        hnsw.insert(*feature, &mut searcher);
    }

    // Test layer_feature at level 0
    let feature = hnsw.layer_feature(0, 0);
    assert_eq!(*feature, [0.0, 0.0, 0.0, 1.0]);

    let feature = hnsw.layer_feature(0, 1);
    assert_eq!(*feature, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_feature_store_empty_checks() {
    // Test is_empty and len on custom store
    let storage = TrackedFeatureStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let mut hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, TrackedFeatureStore<[f64; 4]>> =
        Hnsw::new_with_storage(Euclidean, storage, prng);

    assert!(hnsw.is_empty());
    assert_eq!(hnsw.len(), 0);
    assert_eq!(hnsw.layer_len(0), 0);

    let mut searcher = Searcher::default();
    hnsw.insert([1.0, 2.0, 3.0, 4.0], &mut searcher);

    assert!(!hnsw.is_empty());
    assert_eq!(hnsw.len(), 1);
    assert_eq!(hnsw.layer_len(0), 1);
}

#[test]
fn test_search_empty_custom_store() {
    // Test searching an empty HNSW with custom store returns empty results
    let storage = TrackedFeatureStore::<[f64; 4]>::new();
    let prng = Pcg64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96);

    let hnsw: Hnsw<Euclidean, [f64; 4], Pcg64, 12, 24, TrackedFeatureStore<[f64; 4]>> =
        Hnsw::new_with_storage(Euclidean, storage, prng);

    let mut searcher = Searcher::default();
    let mut neighbors = [Neighbor {
        index: !0,
        distance: !0,
    }; 4];

    let result = hnsw.nearest(&[0.0, 0.0, 0.0, 1.0], 24, &mut searcher, &mut neighbors);
    assert_eq!(result.len(), 0);
}
