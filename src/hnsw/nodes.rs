#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A node in the zero layer
#[derive(Clone, Debug)]
pub struct NeighborNodes<const N: usize> {
    /// The neighbors of this node.
    pub neighbors: [usize; N],
}

impl<const N: usize> NeighborNodes<N> {
    pub fn neighbors(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct Node<const N: usize> {
    /// The node in the zero layer this refers to.
    pub zero_node: usize,
    /// The node in the layer below this one that this node corresponds to.
    pub next_node: usize,
    /// The neighbors in the graph of this node.
    pub neighbors: NeighborNodes<N>,
}

impl<const N: usize> Node<N> {
    pub fn neighbors(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors.neighbors()
    }
}

/// The inbound nodes that are pointing to this node.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct InboundNodes<const N: usize> {
    pub neighbors: SmallVec<[usize; N]>,
}
