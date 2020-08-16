#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use core::fmt;

/// A node in the zero layer
#[derive(Clone)]
pub struct NeighborNodes<const N: usize> {
    /// The neighbors of this node.
    pub neighbors: [u32; N],
}

impl<const N: usize> NeighborNodes<N> {
    pub fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

impl<const N: usize> fmt::Debug for NeighborNodes<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.neighbors.fmt(f)
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct Node<const N: usize> {
    /// The node in the zero layer this refers to.
    pub zero_node: u32,
    /// The node in the layer below this one that this node corresponds to.
    pub next_node: u32,
    /// The neighbors in the graph of this node.
    pub neighbors: NeighborNodes<N>,
}

impl<const N: usize> Node<N> {
    pub fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.neighbors()
    }
}