use super::NeighborNodes;
use core::fmt;
use serde::{
    de::{Error, Expected, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};

impl<const N: usize> Serialize for NeighborNodes<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.neighbors[..].serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for NeighborNodes<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_bytes(NeighborNodesVisitor::<N>)
    }
}

struct NeighborNodesVisitor<const N: usize>;

impl<'de, const N: usize> Visitor<'de> for NeighborNodesVisitor<N> {
    type Value = NeighborNodes<N>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[u32; {}]", N)
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<NeighborNodes<N>, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut neighbors = [!0; N];
        let mut position = 0;

        while let Some(n) = seq.next_element()? {
            if position < N {
                neighbors[position] = n;
                position += 1;
            } else {
                return Err(Error::invalid_length(
                    N + 1,
                    &NeighborNodesExpectedNum::<N>(true),
                ));
            }
        }

        if position != N {
            Err(Error::invalid_length(
                position,
                &NeighborNodesExpectedNum::<N>(false),
            ))
        } else {
            Ok(NeighborNodes { neighbors })
        }
    }
}

struct NeighborNodesExpectedNum<const N: usize>(bool);

impl<const N: usize> Expected for NeighborNodesExpectedNum<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "{} elements was expected; found {}",
            N,
            if self.0 { "too many" } else { "too few" }
        )
    }
}
