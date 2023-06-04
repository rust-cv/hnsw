use core::{
    slice::Iter,
    iter::{TakeWhile, Cloned},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

pub enum Layer<T> {
    Zero,
    NonZero(T),
}

pub trait HasNeighbors<'a, 'b> {
    type NeighborIter: Iterator<Item = usize> + 'a;

    fn get_neighbors(&'b self) -> Self::NeighborIter;
}

/// A node in the zero layer
#[derive(Clone, Debug)]
pub struct NeighborNodes<const N: usize> {
    /// The neighbors of this node.
    pub neighbors: [usize; N],
}

impl<'a, 'b: 'a, const N: usize> HasNeighbors<'a, 'b> for NeighborNodes<N> {
    type NeighborIter = TakeWhile<Cloned<Iter<'a, usize>>, fn(&usize) -> bool>;

    fn get_neighbors(&'b self) -> Self::NeighborIter {
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

impl<'a, 'b: 'a, const N: usize> HasNeighbors<'a, 'b> for Node<N> {
    type NeighborIter = TakeWhile<Cloned<Iter<'a, usize>>, fn(&usize) -> bool>;

    fn get_neighbors(&'b self) -> Self::NeighborIter {
        self.neighbors.get_neighbors()
    }
}

/// The inbound nodes that are pointing to this node.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize), serde(bound = ""))]
pub struct InboundNodes<const N: usize> {
    pub neighbors: SmallVec<[usize; N]>,
}
