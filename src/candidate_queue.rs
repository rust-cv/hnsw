//! This is a special priority queue specifically for 128-bit hamming space searches.
//!
//! This queue works by having 129 vectors, one for each distance. When we find that an internal node
//! achieves a distance of `n` at the least, we place the index of that node into the vector associated
//! with that distance. Any time we take a node off, we place all of its children into the appropriate
//! distance priorities.
//!
//! We maintain the lowest weight vector at any given time in the queue. When a vector runs out,
//! because of the greedy nature of the search algorithm, we are guaranteed that nothing will ever have a distance
//! that lower than the previous candidates. This means we only have to move the lowest weight vector forwards.
//! Also, typically every removal will be constant time since we are incredibly likely to find all the nearest
//! neighbors required before we reach a distance of 64, which is the lowest possible max distance in the root node
//! (distances of the hamming weights 0-64 and 64-128) and the average distance between two random bit strings.
//! The more things in the search, the less likely this becomes. Assuming randomly distributed features, we expect
//! half of the features to have a distance below 64, so it is incredibly likely that all removals are constant time
//! since we will always encounter a removal below or equal to 64.

use std::fmt;

type Distances<T> = [Vec<T>; 129];

#[derive(Clone)]
pub struct CandidateQueue<T> {
    distances: Distances<T>,
    lowest: usize,
}

impl<T> CandidateQueue<T> {
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

    /// This removes the nearest candidate from the queue.
    #[inline]
    pub(crate) fn pop(&mut self) -> Option<(T, u32)> {
        loop {
            if let Some(node) = self.distances[self.lowest].pop() {
                return Some((node, self.lowest as u32));
            } else if self.lowest == 128 {
                return None;
            } else {
                self.lowest += 1;
            }
        }
    }

    /// Inserts a node.
    #[inline]
    pub(crate) fn insert(&mut self, node: T, distance: u32) {
        self.distances[distance as usize].push(node);
    }

    /// Returns the distance if not empty.
    pub(crate) fn distance(&mut self) -> Option<u32> {
        self.distances[self.lowest..]
            .iter()
            .position(|v| !v.is_empty())
            .map(|n| (n + self.lowest) as u32)
    }

    /// Iterator over the items in the queue.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item=&mut T> {
        self.distances.iter_mut().flat_map(|v| v.iter_mut())
    }
}

impl<T> fmt::Debug for CandidateQueue<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<T> Default for CandidateQueue<T> {
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
