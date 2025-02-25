extern crate alloc;

mod hnsw;

use std::cmp::Ordering;

pub use self::hnsw::*;

use ahash::RandomState;
use alloc::{vec, vec::Vec};
use hashbrown::HashSet;
use hnsw::min_max_heap::MinMaxHeap;
use num_traits::Unsigned;
use space::Neighbor;

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
#[derive(Clone, Debug)]
pub struct Searcher<Metric: Ord + Unsigned + Copy> {
    /// The candidates that have been found so far.
    candidates: Vec<Neighbor<Metric>>,
    nearest: MinMaxHeap<NbrNode<Metric>>,
    seen: HashSet<usize, RandomState>,
}

impl<Metric: Ord + Unsigned + Copy> Searcher<Metric> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            nearest: MinMaxHeap::with_capacity(capacity),
            seen: HashSet::with_capacity_and_hasher(capacity, RandomState::with_seeds(0, 0, 0, 0)),
        }
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }

    pub fn get_top_k_nearest(&mut self, k: usize) -> Vec<Neighbor<Metric>> {
        self.nearest
            .pop_min_k(k)
            .into_iter()
            .map(|v| v.nbr)
            .collect()
    }

    pub fn check_push(&mut self, distance: Metric, max_cap: usize) -> bool {
        if self.nearest.len() < max_cap || self.nearest.peek_max().unwrap().nbr.distance > distance
        {
            return true;
        }
        false
    }

    pub fn peek_min(&self) -> Option<&Neighbor<Metric>> {
        self.nearest.peek_min().map(|v| &v.nbr)
    }

    pub fn push(&mut self, nbr: Neighbor<Metric>) {
        self.candidates.push(nbr);
        self.nearest.push(NbrNode { nbr });
    }

    pub fn iter_nearest_index(&self) -> impl Iterator<Item = usize> + '_ {
        self.nearest.iter().map(|v| v.nbr.index)
    }
}

#[derive(Clone, Debug)]
struct NbrNode<Metric: Ord> {
    nbr: Neighbor<Metric>,
}

impl<Metric: Ord> Ord for NbrNode<Metric> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.nbr.distance.cmp(&other.nbr.distance)
    }
}

impl<Metric: Ord> PartialOrd for NbrNode<Metric> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Metric: Ord> PartialEq for NbrNode<Metric> {
    fn eq(&self, other: &Self) -> bool {
        self.nbr.distance == other.nbr.distance
    }
}

impl<Metric: Ord> Eq for NbrNode<Metric> {}

impl<Metric: Ord + Unsigned + Copy> Default for Searcher<Metric> {
    fn default() -> Self {
        Self {
            candidates: vec![],
            nearest: MinMaxHeap::new(),
            seen: HashSet::with_hasher(RandomState::with_seeds(0, 0, 0, 0)),
        }
    }
}
