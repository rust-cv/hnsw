use generic_array::ArrayLength;
pub use packed_simd::{u128x2, u128x4};

pub trait DiscreteDistance {
    type Distances: ArrayLength<Vec<u32>>;

    fn discrete_distance(lhs: &Self, rhs: &Self) -> u32;
}

/// Treats each bit contained in this struct as its own dimension.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct Hamming<T>(pub T);

impl DiscreteDistance for Hamming<u8> {
    type Distances = typenum::U9;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl DiscreteDistance for Hamming<u16> {
    type Distances = typenum::U17;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl DiscreteDistance for Hamming<u32> {
    type Distances = typenum::U33;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl DiscreteDistance for Hamming<u64> {
    type Distances = typenum::U65;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl DiscreteDistance for Hamming<u128> {
    type Distances = typenum::U129;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones()
    }
}

impl DiscreteDistance for Hamming<u128x2> {
    type Distances = typenum::U257;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones().wrapping_sum() as u32
    }
}

impl DiscreteDistance for Hamming<u128x4> {
    type Distances = typenum::U513;

    fn discrete_distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
        (lhs ^ rhs).count_ones().wrapping_sum() as u32
    }
}
