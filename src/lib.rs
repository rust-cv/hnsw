extern crate alloc;

pub mod hnsw;
pub mod metric;
mod nodes;
mod serde_impl;
mod storage;

pub use self::hnsw::*;

use ahash::RandomState;
use hashbrown::HashSet;
use storage::NodeDB;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Params {
    dbpath: std::path::PathBuf,
    ef_construction: usize,
}

impl Params {
    pub fn new(dbpath: String) -> Self {
        Self {
            dbpath: dbpath.into(),
            ef_construction: 0,
        }
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
            dbpath: "/tmp/hnsw.db".into(),
            ef_construction: 400,
        }
    }
}
