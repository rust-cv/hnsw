#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodableFloat {
    pub value: f32,
}

impl From<u32> for EncodableFloat {
    fn from(from: u32) -> EncodableFloat {
        EncodableFloat {
            value: f32::from_bits(from),
        }
    }
}

impl From<EncodableFloat> for u32 {
    fn from(from: EncodableFloat) -> u32 {
        from.value.to_bits()
    }
}

impl Eq for EncodableFloat {}

impl PartialOrd for EncodableFloat {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EncodableFloat {
    fn cmp(&self, rhs: &EncodableFloat) -> core::cmp::Ordering {
        if let Some(o) = self.partial_cmp(rhs) {
            o
        } else {
            core::cmp::Ordering::Equal
        }
    }
}

pub trait Metric<VectorType> {
    type Unit: Into<u32> + From<u32> + Ord + Copy;

    fn distance(&self, lhs: &VectorType, rhs: &VectorType) -> Self::Unit;
}

pub struct SimpleEuclidean;

impl Metric<Vec<f32>> for SimpleEuclidean {
    type Unit = EncodableFloat;
    fn distance(&self, lhs: &Vec<f32>, rhs: &Vec<f32>) -> EncodableFloat {
        let value = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&l, &r)| (l - r).powi(2))
            .sum::<f32>()
            .sqrt();
        EncodableFloat { value }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Neighbor<Unit, Ix = usize> {
    /// Index of the neighbor in the search space.
    pub index: Ix,
    /// The distance of the neighbor from the search feature.
    pub distance: Unit,
}
