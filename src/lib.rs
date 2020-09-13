#![feature(min_const_generics)]
#![no_std]
extern crate alloc;

mod hnsw;

pub use self::hnsw::*;

use alloc::vec::Vec;
use hashbrown::HashSet;
use rustc_hash::FxHasher;
use space::{CandidatesVec, Neighbor};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Params {
    ef_construction: usize,
}

impl Params {
    pub fn new() -> Self {
        Default::default()
    }

    /// This is refered to as `efConstruction` in the paper. This is equivalent to the `ef` parameter passed
    /// to `nearest`, but it is the `ef` used when inserting elements. The higher this is, the more likely the
    /// nearest neighbors in each graph level will be correct, leading to a higher recall rate and speed when
    /// calling `nearest`. This parameter greatly affects the speed of insertion into the HNSW.
    ///
    /// This parameter is probably the only one that in important to tweak.
    ///
    /// Defaults to `400` (overkill for most tasks, but only lower after profiling).
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            ef_construction: 400,
        }
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug, Default)]
pub struct Searcher {
    candidates: Vec<Neighbor>,
    nearest: CandidatesVec,
    seen: HashSet<usize, core::hash::BuildHasherDefault<FxHasher>>,
}

impl Searcher {
    pub fn new() -> Self {
        Default::default()
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}
