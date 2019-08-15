pub use packed_simd::{u128x2, u128x4};

/// This is the primary trait used by the HNSW. This is also implemented for [`FloatingDistance`].
/// If your features have a floating point distance, please implement the distance using [`FloatingDistance`].
/// Implementing [`FloatingDistance`] implements [`Distance`] so long as you satisfy its conditions.
pub trait Distance {
    /// This must compute the distance between two `Self` as a `u32`.
    fn distance(lhs: &Self, rhs: &Self) -> u32;
}

/// Implement this trait when your features have a floating point distance between them. You will take no performance
/// penalty for doing so. Please ensure your distance satisfies the conditions on `floating_distance`.
pub trait FloatingDistance {
    /// This must compute the distance between two `Self` as a `f32`.
    /// The output must not be negative, infinity, or NaN. Subnormal numbers and zero are allowed.
    fn floating_distance(lhs: &Self, rhs: &Self) -> f32;
}

/// This impl requires the float to not be negative, infinite, or NaN.
/// The tradeoff is that it performs equally as well as unsigned integer distance.
impl<T> Distance for T where T: FloatingDistance {
    fn distance(lhs: &Self, rhs: &Self) -> u32 {
        T::floating_distance(lhs, rhs).to_bits()
    }
}

/// Treats each bit contained in this struct as its own dimension and distance is computed as hamming distance.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct Hamming<T>(pub T);

impl Distance for Hamming<&[u8]> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        // TODO: This generates pretty sub-optimal code.
        lhs.iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones())
            .sum::<u32>()
    }
}

impl Distance for Hamming<u8> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl Distance for Hamming<u16> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl Distance for Hamming<u32> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl Distance for Hamming<u64> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl Distance for Hamming<u128> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl Distance for Hamming<u128x2> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones().wrapping_sum() as u32
    }
}

impl Distance for Hamming<u128x4> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones().wrapping_sum() as u32
    }
}

impl Distance for Hamming<[u128x4; 2]> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        lhs.iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones().wrapping_sum() as u32)
            .sum::<u32>()
    }
}

impl Distance for Hamming<[u128x4; 4]> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        lhs.iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones().wrapping_sum() as u32)
            .sum::<u32>()
    }
}

impl Distance for Hamming<[u128x4; 8]> {
    fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        lhs.iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones().wrapping_sum() as u32)
            .sum::<u32>()
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
