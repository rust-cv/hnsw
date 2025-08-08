#![no_std]
extern crate alloc;

mod hnsw;

pub use self::hnsw::*;

use ahash::RandomState;
use alloc::{vec, vec::Vec};
use hashbrown::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Params {
    ef_construction: usize,
    ef_search: usize,
}

impl Params {
    pub fn new() -> Self {
        Default::default()
    }

    /// This is refered to as `efConstruction` in the paper. This is equivalent to the `ef` parameter passed
    /// to `nearest`, but it is the `ef` used when inserting elements. The higher this is, the more likely the
    /// nearest neighbors in each graph level will be correct, leading to a higher recall rate and lower speed when
    /// calling `nearest`. This parameter greatly affects the speed of insertion into the HNSW.
    ///
    /// This parameter is probably the only one that is important to tweak.
    ///
    /// Defaults to `400` (only adjust after profiling).
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// This is a parameter that is used to set `ef` from the paper. This value is added to the number of items
    /// requested to set the `ef` parameter passed to `nearest` when searching. The higher this is, the more likely
    /// the nearest neighbors in each graph level will be correct, leading to a higher recall rate and lower speed
    /// when calling `nearest`. This parameter greatly affects the speed of search in HNSW.
    ///
    /// Defaults to `16` (only adjust after profiling).
    pub fn ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            ef_construction: 400,
            ef_search: 16,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Neighbor<Unit> {
    pub index: usize,
    pub distance: Unit,
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
pub struct Searcher<Unit> {
    candidates: Vec<Neighbor<Unit>>,
    nearest: Vec<Neighbor<Unit>>,
    seen: HashSet<usize, RandomState>,
}

impl<Unit> Searcher<Unit> {
    pub fn new() -> Self {
        Self {
            candidates: vec![],
            nearest: vec![],
            seen: HashSet::with_hasher(RandomState::with_seeds(0, 0, 0, 0)),
        }
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<Unit> Default for Searcher<Unit> {
    fn default() -> Self {
        Self::new()
    }
}
