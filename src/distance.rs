pub use packed_simd::{f32x16, f32x2, f32x4, f32x8, u8x16, u8x2, u8x32, u8x4, u8x64, u8x8};
#[cfg(feature = "serde-impl")]
use serde::{Deserialize, Serialize};

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
impl<T> Distance for T
where
    T: FloatingDistance,
{
    fn distance(lhs: &Self, rhs: &Self) -> u32 {
        T::floating_distance(lhs, rhs).to_bits()
    }
}

/// Treats each bit contained in this struct as its own dimension and distance is computed as hamming distance.
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
#[cfg_attr(feature = "serde-impl", derive(Serialize, Deserialize))]
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

impl Distance for Hamming<Vec<u8>> {
    fn distance(Self(lhs): &Self, Self(rhs): &Self) -> u32 {
        // TODO: This generates pretty sub-optimal code.
        lhs.iter()
            .zip(rhs)
            .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones())
            .sum::<u32>()
    }
}

macro_rules! hamming_native_impl {
    ($x:ty) => {
        impl Distance for Hamming<$x> {
            fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
                (lhs ^ rhs).count_ones()
            }
        }
    };
}

hamming_native_impl!(u8);
hamming_native_impl!(u16);
hamming_native_impl!(u32);
hamming_native_impl!(u64);
hamming_native_impl!(u128);

macro_rules! hamming_u8_simd_impl {
    ($x:ty) => {
        impl Distance for Hamming<$x> {
            fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
                (lhs ^ rhs).count_ones().wrapping_sum() as u32
            }
        }

        impl From<&[u8]> for Hamming<$x> {
            fn from(a: &[u8]) -> Self {
                Self(<$x>::from_slice_unaligned(a))
            }
        }
    };
}

hamming_u8_simd_impl!(u8x2);
hamming_u8_simd_impl!(u8x4);
hamming_u8_simd_impl!(u8x8);
hamming_u8_simd_impl!(u8x16);
hamming_u8_simd_impl!(u8x32);
hamming_u8_simd_impl!(u8x64);

macro_rules! hamming_u8x64_simd_array_impl {
    ($x:expr) => {
        impl Distance for Hamming<[u8x64; $x]> {
            fn distance(&Self(lhs): &Self, &Self(rhs): &Self) -> u32 {
                lhs.iter()
                    .zip(&rhs)
                    .map(|(&lhs, &rhs)| (lhs ^ rhs).count_ones().wrapping_sum() as u32)
                    .sum::<u32>()
            }
        }

        impl From<[u8; 64 * $x]> for Hamming<[u8x64; $x]> {
            fn from(a: [u8; 64 * $x]) -> Self {
                a.into()
            }
        }

        impl From<&[u8]> for Hamming<[u8x64; $x]> {
            fn from(a: &[u8]) -> Self {
                let mut simd = [u8x64::splat(0); $x];
                for i in 0..$x {
                    simd[i] = u8x64::from_slice_unaligned(&a[i * 64..(i + 1) * 64]);
                }
                Self(simd)
            }
        }

        impl Into<[u8; 64 * $x]> for Hamming<[u8x64; $x]> {
            fn into(self) -> [u8; 64 * $x] {
                let mut a = [0; 64 * $x];
                for (simd_ix, chunk) in a.chunks_exact_mut(64).enumerate() {
                    for (ix, n) in chunk.iter_mut().enumerate() {
                        *n = self.0[simd_ix].extract(ix);
                    }
                }
                a
            }
        }
    };
}

hamming_u8x64_simd_array_impl!(1);
hamming_u8x64_simd_array_impl!(2);
hamming_u8x64_simd_array_impl!(4);
hamming_u8x64_simd_array_impl!(8);
hamming_u8x64_simd_array_impl!(16);
hamming_u8x64_simd_array_impl!(32);

/// Any list, vector, etc of floats wrapped in `Euclidean` is to be treated as having euclidean distance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-impl", derive(Serialize, Deserialize))]
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

impl FloatingDistance for Euclidean<Vec<f32>> {
    fn floating_distance(Euclidean(lhs): &Self, Euclidean(rhs): &Self) -> f32 {
        assert_eq!(lhs.len(), rhs.len());
        lhs.iter()
            .zip(rhs)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
    }
}

impl FloatingDistance for Euclidean<f32> {
    fn floating_distance(&Euclidean(lhs): &Self, &Euclidean(rhs): &Self) -> f32 {
        (lhs - rhs).abs()
    }
}

macro_rules! euclidean_f32_simd_impl {
    ($x:ty) => {
        impl FloatingDistance for Euclidean<$x> {
            fn floating_distance(&Euclidean(lhs): &Self, &Euclidean(rhs): &Self) -> f32 {
                let diff = lhs - rhs;
                (diff * diff).sum()
            }
        }
    };
}

euclidean_f32_simd_impl!(f32x2);
euclidean_f32_simd_impl!(f32x4);
euclidean_f32_simd_impl!(f32x8);
euclidean_f32_simd_impl!(f32x16);

macro_rules! euclidean_f32x16_simd_array_impl {
    ($x:expr) => {
        impl FloatingDistance for Euclidean<[f32x16; $x]> {
            fn floating_distance(&Euclidean(lhs): &Self, &Euclidean(rhs): &Self) -> f32 {
                lhs.iter()
                    .zip(rhs.iter())
                    .map(|(&a, &b)| {
                        let diff = a - b;
                        (diff * diff).sum()
                    })
                    .sum::<f32>()
            }
        }

        impl From<[f32; 16 * $x]> for Euclidean<[f32x16; $x]> {
            fn from(a: [f32; 16 * $x]) -> Self {
                a.into()
            }
        }

        impl From<&[f32]> for Euclidean<[f32x16; $x]> {
            fn from(a: &[f32]) -> Self {
                let mut simd = [f32x16::splat(0.0); $x];
                for i in 0..$x {
                    simd[i] = f32x16::from_slice_unaligned(&a[i * 16..(i + 1) * 16]);
                }
                Self(simd)
            }
        }

        impl Into<[f32; 16 * $x]> for Euclidean<[f32x16; $x]> {
            fn into(self) -> [f32; 16 * $x] {
                let mut a = [0.0; 16 * $x];
                for (simd_ix, chunk) in a.chunks_exact_mut(64).enumerate() {
                    for (ix, n) in chunk.iter_mut().enumerate() {
                        *n = self.0[simd_ix].extract(ix);
                    }
                }
                a
            }
        }
    };
}

euclidean_f32x16_simd_array_impl!(1);
euclidean_f32x16_simd_array_impl!(2);
euclidean_f32x16_simd_array_impl!(3);
euclidean_f32x16_simd_array_impl!(4);
euclidean_f32x16_simd_array_impl!(5);
euclidean_f32x16_simd_array_impl!(6);
euclidean_f32x16_simd_array_impl!(7);
euclidean_f32x16_simd_array_impl!(8);
euclidean_f32x16_simd_array_impl!(9);
euclidean_f32x16_simd_array_impl!(10);
euclidean_f32x16_simd_array_impl!(11);
euclidean_f32x16_simd_array_impl!(12);
euclidean_f32x16_simd_array_impl!(13);
euclidean_f32x16_simd_array_impl!(14);
euclidean_f32x16_simd_array_impl!(15);
euclidean_f32x16_simd_array_impl!(16);
euclidean_f32x16_simd_array_impl!(32);
euclidean_f32x16_simd_array_impl!(64);
euclidean_f32x16_simd_array_impl!(128);
euclidean_f32x16_simd_array_impl!(256);
