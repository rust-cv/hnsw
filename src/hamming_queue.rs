//! This is a special priority queue specifically for 128-bit hamming weight searches.
//!
//! This queue works by having 129 vectors, one for each distance. When we find that an internal node
//! achieves a distance of `n` at the least, we place the index of that node into the vector associated
//! with that distance. Any time we take a node off, we place all of its children into the appropriate
//! distance priorities.
//!
//! We maintain the lowest weight vector at any given time in the queue. When a vector runs out,
//! because of the greedy nature of the search algorithm, we are guaranteed that nothing will ever have a distance
//! that low again, since the sum of the distance of bit substrings can only be higher than the distance of
//! their parents. This means we only have to move the lowest weight vector forwards. Also, typically every
//! removal will be constant time since we are incredibly likely to find all the nearest neighbor's required
//! before we reach a distance of 64, which is the lowest possible max distance in the root node (distances
//! of the hamming weights 0-64 and 64-128). The more things in the hamming weight tree, the less likely
//! this becomes. Assuming randomly distributed features, we expect half of the features to have a distance
//! below 64, so it is incredibly likely that all removals are constant time since we will always encounter
//! a removal below or equal to 64.

use std::fmt;

type Distances<T> = [Vec<(T, u8)>; 129];
type NodeEntry<T> = (u32, T, u8);

#[derive(Clone)]
pub struct NodeQueue<T> {
    distances: Distances<T>,
    lowest: usize,
}

impl<T> NodeQueue<T> {
    /// Takes all the entries in the root node (level 0) and adds them to the queue.
    ///
    /// This is passed the (distance, tp, node).
    pub fn new() -> Self {
        Default::default()
    }

    /// This allows the queue to be cleared so that we don't need to reallocate memory.
    pub(crate) fn clear(&mut self) {
        for v in self.distances.iter_mut() {
            v.clear();
        }
        self.lowest = 0;
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<NodeEntry<T>> {
        loop {
            if let Some((node, level)) = self.distances[self.lowest].pop() {
                return Some((self.lowest as u32, node, level));
            } else if self.lowest == 128 {
                return None;
            } else {
                self.lowest += 1;
            }
        }
    }

    /// Takes an iterator over (distance, tp, node, level)
    #[inline]
    pub(crate) fn add_one(&mut self, (distance, node, level): NodeEntry<T>) {
        self.distances[distance as usize].push((node, level));
    }

    /// Returns the distance if not empty.
    pub(crate) fn distance(&mut self) -> Option<u32> {
        self.distances[self.lowest..]
            .iter()
            .position(|v| !v.is_empty())
            .map(|n| (n + self.lowest) as u32)
    }
}

impl<T> fmt::Debug for NodeQueue<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<T> Default for NodeQueue<T> {
    fn default() -> Self {
        Self {
            distances: [
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
            lowest: 0,
        }
    }
}
