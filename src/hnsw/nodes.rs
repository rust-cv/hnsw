use core::{
    iter::{Cloned, TakeWhile},
    slice::Iter,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
pub struct ZeroNode<const N: usize> {
    /// The neighbors of this node.
    pub neighbors: [usize; N],
}

impl<'a, 'b: 'a, const N: usize> HasNeighbors<'a, 'b> for ZeroNode<N> {
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
    pub neighbors: ZeroNode<N>,
}

impl<'a, 'b: 'a, const N: usize> HasNeighbors<'a, 'b> for Node<N> {
    type NeighborIter = TakeWhile<Cloned<Iter<'a, usize>>, fn(&usize) -> bool>;

    fn get_neighbors(&'b self) -> Self::NeighborIter {
        self.neighbors.get_neighbors()
    }
}
