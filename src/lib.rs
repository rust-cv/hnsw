mod candidate_queue;
mod nearest_queue;

use candidate_queue::*;
use nearest_queue::*;
use rand_pcg::Pcg64;
use std::collections::HashSet;

const M: usize = 12;
const M_MAX: usize = M;
const M_MAX0: usize = M * 2;
const NUM_PRESERVED_CANDIDATES: usize = 1;

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
    /// This needs to create resonably random outputs to determine the levels of insertions.
    prng: Pcg64,
}

/// A node in the zero layer
#[derive(Clone, Debug)]
struct ZeroNode {
    /// The neighbors of this node.
    neighbors: [u32; M_MAX0],
}

impl ZeroNode {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
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

impl Node {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug, Default)]
pub struct Searcher {
    candidates: CandidateQueue<u32>,
    nearest: NearestQueue<u32>,
    seen: HashSet<u32>,
}

impl Searcher {
    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.reset(0);
        self.seen.clear();
    }
}

impl HNSW {
    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: u128, searcher: &mut Searcher) -> u32 {
        // Get the level of this feature.
        let level = self.random_level();

        self.initialize_searcher(q, searcher);

        // Start from the current top layer and connect it to its nearest neighbors.
        for ix in (0..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_layer(q, searcher, &self.layers[ix]);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(q, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher, if ix == 0 { M_MAX0 } else { M });
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(q, searcher);
        self.create_node(q, &searcher.nearest, 0);
        // Add the feature to the zero layer.
        self.features.push(q);

        // Add all level vectors needed to be able to add this level.
        let zero_node = (self.zero.len() - 1) as u32;
        while self.layers.len() < level {
            let node = Node {
                zero_node,
                next_node: self
                    .layers
                    .last()
                    .map(|l| (l.len() - 1) as u32)
                    .unwrap_or(zero_node),
                neighbors: [!0; M],
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'a>(
        &self,
        q: u128,
        searcher: &mut Searcher,
        dest: &'a mut [u32],
    ) -> &'a mut [u32] {
        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher);

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_layer(q, searcher, layer);
            self.lower_search(layer, searcher, if ix == 0 { M_MAX0 } else { M });
        }

        self.search_zero_layer(q, searcher);

        searcher.nearest.fill_slice(dest)
    }

    pub fn feature(&self, item: u32) -> u128 {
        self.features[item as usize]
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_layer(&self, q: u128, searcher: &mut Searcher, layer: &[Node]) {
        while let Some((node, _)) = searcher.candidates.pop() {
            for neighbor in layer[node as usize].neighbors() {
                let neighbor_node = &layer[neighbor as usize];
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor_node.zero_node) {
                    // Compute the distance of this neighbor.
                    let distance =
                        (q ^ self.features[neighbor_node.zero_node as usize]).count_ones();
                    // Attempt to insert into nearest queue.
                    if searcher.nearest.insert(neighbor, distance) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.insert(neighbor, distance);
                    }
                }
            }
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&self, q: u128, searcher: &mut Searcher) {
        while let Some((node, _)) = searcher.candidates.pop() {
            for neighbor in self.zero[node as usize].neighbors() {
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor) {
                    // Compute the distance of this neighbor.
                    let distance = (q ^ self.features[neighbor as usize]).count_ones();
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
        // Only preserve some of the candidates. The original paper's algorithm uses `1` every time,
        // but for benchmarking purposes we will use a constant. See Algorithm 5 line 5 of the paper.
        // The paper makes no further comment on why `1` was chosen.
        searcher.nearest.set_size(NUM_PRESERVED_CANDIDATES);
        // Look through all the nearest neighbors from the last layer.
        for (node, distance) in searcher.nearest.iter_mut() {
            // Update the node to the next layer.
            *node = layer[*node as usize].next_node;
            // Insert the indices of those nearest neighbors into the candidates for the next layer.
            searcher.candidates.insert(*node, distance);
        }
        // Set the capacity on the nearest to `m`.
        searcher.nearest.set_capacity(m);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&self, q: u128, searcher: &mut Searcher) {
        // Clear the searcher.
        searcher.clear();
        // Add the entry point.
        searcher
            .candidates
            .insert(0, (q ^ self.entry_feature()).count_ones());
        searcher
            .nearest
            .insert(0, (q ^ self.entry_feature()).count_ones());
        searcher.seen.insert(
            self.layers
                .last()
                .map(|layer| layer[0].zero_node)
                .unwrap_or(0),
        );
    }

    /// Gets the entry point's feature.
    fn entry_feature(&self) -> u128 {
        if let Some(last_layer) = self.layers.last() {
            self.features[last_layer[0].zero_node as usize]
        } else {
            self.features[0]
        }
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        use rand_distr::{Distribution, Standard};
        let uniform: f32 = Standard.sample(&mut self.prng);
        (-uniform.ln() * (M as f32).ln().recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: u128, nearest: &NearestQueue<u32>, layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors = [!0; M_MAX0];
            nearest.fill_slice(&mut neighbors);
            let node = ZeroNode { neighbors };
            for neighbor in node.neighbors() {
                self.add_neighbor(q, new_index as u32, neighbor, layer);
            }
            self.zero.push(node);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors = [!0; M_MAX];
            nearest.fill_slice(&mut neighbors);
            let node = Node {
                zero_node: self.zero.len() as u32,
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                } as u32,
                neighbors,
            };
            for neighbor in node.neighbors() {
                self.add_neighbor(q, new_index as u32, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// Attempts to add a neighbor to a target node.
    fn add_neighbor(&mut self, q: u128, node_ix: u32, target_ix: u32, layer: usize) {
        // Get the feature for the target and get the neighbor slice for the target.
        // This is different for the zero layer.
        let (target_feature, target_neighbors) = if layer == 0 {
            (
                self.features[target_ix as usize],
                &self.zero[target_ix as usize].neighbors[..],
            )
        } else {
            let target = &self.layers[layer - 1][target_ix as usize];
            (
                self.features[target.zero_node as usize],
                &target.neighbors[..],
            )
        };

        // Get the worst neighbor of this node currently.
        let (worst_ix, worst_distance) = target_neighbors
            .iter()
            .enumerate()
            .map(|(ix, &n)| {
                // Compute the distance to be higher than possible if the neighbor is not filled yet so its always filled.
                let distance = if n == !0 {
                    129
                } else {
                    // Compute the distance. The feature is looked up differently for the zero layer.
                    (target_feature
                        ^ self.features[if layer == 0 {
                            n as usize
                        } else {
                            self.layers[layer - 1][n as usize].zero_node as usize
                        }])
                    .count_ones()
                };
                (ix, distance)
            })
            .min_by_key(|&(_, distance)| distance)
            .unwrap();

        // If this is better than the worst, insert it in the worst's place.
        // This is also different for the zero layer.
        if (q ^ target_feature).count_ones() < worst_distance {
            if layer == 0 {
                self.zero[target_ix as usize].neighbors[worst_ix] = node_ix;
            } else {
                self.layers[layer - 1][target_ix as usize].neighbors[worst_ix] = node_ix;
            }
        }
    }
}
