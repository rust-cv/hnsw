use crate::*;

use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;
use rustc_hash::FxHasher;
use std::collections::HashSet;

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement `FloatingDistance` to get implementations.
#[derive(Clone)]
pub struct HNSW<
    T,
    M: ArrayLength<u32> = typenum::U12,
    M0: ArrayLength<u32> = typenum::U24,
    R = Pcg64,
> {
    /// Contains the zero layer.
    zero: Vec<ZeroNode<M0>>,
    /// Contains the features of the zero layer.
    /// These are stored separately to allow SIMD speedup in the future by
    /// grouping small worlds of features together.
    features: Vec<T>,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Node<M>>>,
    /// This needs to create resonably random outputs to determine the levels of insertions.
    prng: R,
    /// The parameters for the HNSW.
    params: Params,
}

/// A node in the zero layer
#[derive(Clone)]
struct ZeroNode<N: ArrayLength<u32>> {
    /// The neighbors of this node.
    neighbors: GenericArray<u32, N>,
}

impl<N: ArrayLength<u32>> ZeroNode<N> {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug)]
struct Node<N: ArrayLength<u32>> {
    /// The node in the zero layer this refers to.
    zero_node: u32,
    /// The node in the layer below this one that this node corresponds to.
    next_node: u32,
    /// The neighbors in the graph of this node.
    neighbors: GenericArray<u32, N>,
}

impl<N: ArrayLength<u32>> Node<N> {
    fn neighbors<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.neighbors.iter().cloned().take_while(|&n| n != !0)
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug, Default)]
pub struct Searcher {
    candidates: Candidates,
    nearest: FixedCandidates,
    seen: HashSet<u32, std::hash::BuildHasherDefault<FxHasher>>,
}

impl Searcher {
    pub fn new() -> Self {
        Default::default()
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> HNSW<T, M, M0, R>
where
    R: RngCore + SeedableRng,
{
    /// Creates a new HNSW with a PRNG which is default seeded to produce deterministic behavior.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new HNSW with a default seeded PRNG and with the specified params.
    pub fn new_params(params: Params) -> Self {
        Self {
            params,
            ..Default::default()
        }
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> HNSW<T, M, M0, R>
where
    R: RngCore,
    T: FloatingDistance,
{
    /// Creates a HNSW with the passed `prng`.
    pub fn new_prng(prng: R) -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params: Default::default(),
        }
    }

    /// Creates a HNSW with the passed `params` and `prng`.
    pub fn new_params_and_prng(params: Params, prng: R) -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params,
        }
    }

    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher) -> u32 {
        // Get the level of this feature.
        let level = self.random_level();

        // If this is empty, none of this will work, so just add it manually.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(ZeroNode {
                neighbors: std::iter::repeat(!0).collect(),
            });
            self.features.push(q);

            // Add all the layers its in.
            while self.layers.len() < level {
                // It's always index 0 with no neighbors since its the first feature.
                let node = Node {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: std::iter::repeat(!0).collect(),
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(
            &q,
            searcher,
            if level >= self.layers.len() {
                self.params.ef_construction
            } else {
                1
            },
        );

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_layer(&q, searcher, &self.layers[ix]);
            // Then lower the search only after we create the node.
            self.lower_search(
                &self.layers[ix],
                searcher,
                if ix == level {
                    self.params.ef_construction
                } else {
                    1
                },
            );
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..std::cmp::min(level, self.layers.len())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_layer(&q, searcher, &self.layers[ix]);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(&q, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher, self.params.ef_construction);
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&q, searcher);
        self.create_node(&q, &searcher.nearest, 0);
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
                neighbors: std::iter::repeat(!0).collect(),
            };
            self.layers.push(vec![node]);
        }
        zero_node
    }

    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    /// `ef` is the candidate pool size. `ef` can be increased to get better recall at the expense of speed.
    /// If `ef` is less than `dest.len()` then `dest` will only be filled with `ef` elements.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'a>(
        &self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher,
        dest: &'a mut [u32],
    ) -> &'a mut [u32] {
        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher, if self.layers.is_empty() { ef } else { 1 });

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_layer(q, searcher, layer);
            self.lower_search(layer, searcher, if ix == 0 { ef } else { 1 });
        }

        self.search_zero_layer(q, searcher);

        searcher.nearest.fill_slice(dest)
    }

    pub fn feature(&self, item: u32) -> &T {
        &self.features[item as usize]
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_layer(&self, q: &T, searcher: &mut Searcher, layer: &[Node<M>]) {
        while let Some((_, node)) = searcher.candidates.pop() {
            for neighbor in layer[node as usize].neighbors() {
                let neighbor_node = &layer[neighbor as usize];
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor_node.zero_node) {
                    // Compute the distance of this neighbor.
                    let distance =
                        T::floating_distance(q, &self.features[neighbor_node.zero_node as usize]);
                    // Attempt to insert into nearest queue.
                    if searcher.nearest.push(distance, neighbor) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.push(distance, neighbor);
                    }
                }
            }
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&self, q: &T, searcher: &mut Searcher) {
        while let Some((_, node)) = searcher.candidates.pop() {
            for neighbor in self.zero[node as usize].neighbors() {
                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(neighbor) {
                    // Compute the distance of this neighbor.
                    let distance = T::floating_distance(q, &self.features[neighbor as usize]);
                    // Attempt to insert into nearest queue.
                    if searcher.nearest.push(distance, neighbor) {
                        // If inserting into the nearest queue was sucessful, we want to add this node to the candidates.
                        searcher.candidates.push(distance, neighbor);
                    }
                }
            }
        }
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search(&self, layer: &[Node<M>], searcher: &mut Searcher, m: usize) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Only preserve the best candidate. The original paper's algorithm uses `1` every time.
        // See Algorithm 5 line 5 of the paper. The paper makes no further comment on why `1` was chosen.
        let (distance, node) = searcher.nearest.pop().unwrap();
        searcher.nearest.clear();
        // Set the capacity on the nearest to `m`.
        searcher.nearest.set_cap(m);
        // Update the node to the next layer.
        let new_node = layer[node as usize].next_node;
        // Insert the index of the nearest neighbor into the nearest pool for the next layer.
        searcher.nearest.push(distance, new_node);
        // Insert the index into the candidate pool as well.
        searcher.candidates.push(distance, new_node);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&self, q: &T, searcher: &mut Searcher, cap: usize) {
        // Clear the searcher.
        searcher.clear();
        searcher.nearest.set_cap(cap);
        // Add the entry point.
        let entry_distance = T::floating_distance(q, self.entry_feature());
        searcher.candidates.push(entry_distance, 0);
        searcher.nearest.push(entry_distance, 0);
        searcher.seen.insert(
            self.layers
                .last()
                .map(|layer| layer[0].zero_node)
                .unwrap_or(0),
        );
    }

    /// Gets the entry point's feature.
    fn entry_feature(&self) -> &T {
        if let Some(last_layer) = self.layers.last() {
            &self.features[last_layer[0].zero_node as usize]
        } else {
            &self.features[0]
        }
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        use rand_distr::{Distribution, Standard};
        let uniform: f32 = Standard.sample(&mut self.prng);
        (-uniform.ln() * (M::to_usize() as f32).ln().recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: &T, nearest: &FixedCandidates, layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors: GenericArray<u32, M0> = std::iter::repeat(!0).collect();
            nearest.fill_slice(&mut neighbors);
            let node = ZeroNode { neighbors };
            for neighbor in node.neighbors() {
                self.add_neighbor(q, new_index as u32, neighbor, layer);
            }
            self.zero.push(node);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors: GenericArray<u32, M> = std::iter::repeat(!0).collect();
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
    fn add_neighbor(&mut self, q: &T, node_ix: u32, target_ix: u32, layer: usize) {
        // Get the feature for the target and get the neighbor slice for the target.
        // This is different for the zero layer.
        let (target_feature, target_neighbors) = if layer == 0 {
            (
                &self.features[target_ix as usize],
                &self.zero[target_ix as usize].neighbors[..],
            )
        } else {
            let target = &self.layers[layer - 1][target_ix as usize];
            (
                &self.features[target.zero_node as usize],
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
                    std::f32::MAX
                } else {
                    // Compute the distance. The feature is looked up differently for the zero layer.
                    T::floating_distance(
                        target_feature,
                        &self.features[if layer == 0 {
                            n as usize
                        } else {
                            self.layers[layer - 1][n as usize].zero_node as usize
                        }],
                    )
                };
                (ix, distance)
            })
            // This was done instead of max_by_key because min_by_key takes the first equally bad element.
            .min_by_key(|&(_, distance)| {
                let cast_distance: u32 = unsafe { std::mem::transmute(distance) };
                // This inverts the order for the min_by_key.
                !cast_distance
            })
            .unwrap();

        // If this is better than the worst, insert it in the worst's place.
        // This is also different for the zero layer.
        if T::floating_distance(q, target_feature) < worst_distance {
            if layer == 0 {
                self.zero[target_ix as usize].neighbors[worst_ix] = node_ix;
            } else {
                self.layers[layer - 1][target_ix as usize].neighbors[worst_ix] = node_ix;
            }
        }
    }
}

impl<T, M: ArrayLength<u32>, M0: ArrayLength<u32>, R> Default for HNSW<T, M, M0, R>
where
    R: SeedableRng,
{
    fn default() -> Self {
        Self {
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params: Params::new(),
        }
    }
}
