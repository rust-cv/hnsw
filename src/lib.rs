mod candidate_queue;
mod nearest_queue;

use candidate_queue::*;
use nearest_queue::*;
use rand_distr::{Distribution, Standard};
use rand_pcg::Pcg64;
use std::collections::HashSet;

const M: usize = 12;
const M_MAX: usize = M;
const M_MAX0: usize = M * 2;

// Can't do this as a const because `ln` and `recip` are not const functions yet.
fn ml() -> f32 {
    (M as f32).ln().recip()
}

/// This provides a HNSW implementation for 128-bit hamming space.
#[derive(Clone, Debug)]
pub struct HNSW {
    /// Contains the zero layer.
    zero: Vec<ZeroNode>,
    /// Contains the features of the zero layer.
    /// These are stored separately to allow SIMD speedup in the future by
    /// grouping small worlds of features together.
    features: Vec<u128>,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Node>>,
    /// The features which are referenced by the NSWs.
    /// This is the index into the highest-level world currently existing that we will start searching from.
    enter_index: u32,
    /// This needs to create resonably random outputs to determine the levels of insertions.
    prng: Pcg64,
}

/// A node in the zero layer
#[derive(Clone, Debug)]
struct ZeroNode {
    /// The actual binary feature.
    ///
    /// TODO: This is bad because each feature has to be looked up individually with random access.
    /// We might be able to store the features of neighbors within the neighbors array to speed up the search, but this
    /// will come at the cost of increased memory consumption. With small binary features this would not be an issue.
    feature: u128,
    /// The neighbors of this node.
    neighbors: [u32; M_MAX0],
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
struct Node {
    /// The node in the zero layer this refers to.
    zero_node: u32,
    /// The node in the layer below this one that this node corresponds to.
    next_node: u32,
    /// The neighbors in the graph of this node.
    neighbors: [u32; M_MAX],
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
pub struct Searcher {
    candidates: CandidateQueue<u32>,
    nearest: NearestQueue<u32>,
    seen: HashSet<u32>,
}

impl HNSW {
    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, feature: u128) -> u32 {
        let uniform: f32 = Standard.sample(&mut self.prng);
        let level = (uniform * ml()) as usize;
        // If the level chosen is higher than all other levels, we want to set this to be the enter point.
        if level > self.levels() {
            unimplemented!();
        }
        (self.features.len() - 1) as u32
    }

    /// Does a k-NN search where `k` is the number of nearest neighbors.
    pub fn nearest(&self, feature: u128, k: usize) -> NearestQueue<u128> {
        let feature_heap = NearestQueue::new();
        let mut node_queue: CandidateQueue<u32> = CandidateQueue::new();

        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() {
            return feature_heap;
        }

        // Start by adding the entry point to the node priority queue.
        node_queue.insert(
            self.enter_index,
            (feature ^ self.entry_feature()).count_ones(),
        );

        feature_heap
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a given layer.
    fn search_layer(&self, q: u128, searcher: &mut Searcher, layer: &[Node]) {
        while let Some((node, _)) = searcher.candidates.pop() {
            for neighbor in layer[node as usize]
                .neighbors
                .iter()
                .cloned()
                .take_while(|&n| n != !0)
            {
                let neighbor_node = &layer[neighbor as usize];
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor_node.zero_node) {
                    // Compute the distance of this neighbor.
                    let distance = (self.node_feature(&layer[neighbor as usize]) ^ q).count_ones();
                    // Attempt to insert into nearest queue.
                    if searcher.nearest.insert(neighbor, distance) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.insert(neighbor, distance);
                    }
                }
            }
        }
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search(&self, layer: &[Node], searcher: &mut Searcher, m: usize) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Look through all the nearest neighbors from the last layer.
        for (node, distance) in searcher.nearest.drain() {
            // Insert the indices of those nearest neighbors into the candidates for the next layer.
            searcher
                .candidates
                .insert(layer[node as usize].next_node, distance);
        }
        // Reset the nearest neighbors.
        searcher.nearest.reset(M);
    }

    /// Get a feature from a node.
    fn node_feature(&self, node: &Node) -> u128 {
        self.zero[node.zero_node as usize].feature
    }

    /// Gets the entry point's feature.
    fn entry_feature(&self) -> u128 {
        if let Some(last_layer) = self.layers.last() {
            self.node_feature(&last_layer[self.enter_index as usize])
        } else {
            self.zero[self.enter_index as usize].feature
        }
    }

    /// Get the current number of levels.
    fn levels(&self) -> usize {
        1 + self.layers.len()
    }
}
