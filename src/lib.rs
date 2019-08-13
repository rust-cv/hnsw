mod candidates;
mod discrete_distance;
mod discrete_hnsw;
mod hnsw;
pub use self::hnsw::*;
pub(crate) use candidates::*;
pub use discrete_distance::*;
pub use discrete_hnsw::*;

pub use generic_array::{typenum, ArrayLength, GenericArray};

pub trait FloatingDistance {
    fn floating_distance(lhs: &Self, rhs: &Self) -> f32;
}

#[derive(Copy, Clone, Debug)]
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

/// Any list, vector, etc of floats wrapped in `Euclidean` is to be treated as having euclidean distance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Euclidean<T>(pub T);

impl FloatingDistance for Euclidean<&[f32]> {
    fn floating_distance(&Euclidean(lhs): &Self, &Euclidean(rhs): &Self) -> f32 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
    }
}
