mod discrete_distance;
mod discrete_hnsw;
pub use discrete_distance::*;
pub use discrete_hnsw::*;

pub use generic_array::{typenum, ArrayLength, GenericArray};

#[derive(Copy, Clone, Debug)]
pub struct Params {
    num_preserved: usize,
    num_preserved_construction: usize,
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

    /// This doesn't have a name in the paper, nor is it a part of any algorithm in the paper. In the paper,
    /// every time a search is performed in a level, only the nearest neighbor is preserved when going down
    /// to the next level. This allows more search results to be included in the next level, leading to a
    /// more accurate search. My basic findings thus far are that this has a negligable effect on recall,
    /// but has a negative effect on performance.
    ///
    /// Defaults to `1`.
    pub fn num_preserved(mut self, num_preserved: usize) -> Self {
        self.num_preserved = num_preserved;
        self
    }

    /// This doesn't have a name in the paper, nor is it a part of any algorithm in the paper.
    pub fn num_preserved_construction(mut self, num_preserved_construction: usize) -> Self {
        self.num_preserved_construction = num_preserved_construction;
        self
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            num_preserved: 1,
            num_preserved_construction: 1,
            ef_construction: 400,
        }
    }
}
