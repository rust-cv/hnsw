use super::nodes::{HasNeighbors, Layer};
use crate::hnsw::nodes::{NeighborNodes, Node};
use crate::*;
use alloc::{vec, vec::Vec};
use num_traits::Zero;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use space::{Knn, KnnPoints, Metric, Neighbor};

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement [`space::Metric`] to get implementations.
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "Met: Serialize, T: Serialize, R: Serialize",
        deserialize = "Met: Deserialize<'de>, T: Deserialize<'de>, R: Deserialize<'de>"
    ))
)]
pub struct Hnsw<Met, T, R, const M: usize, const M0: usize> {
    /// Contains the space metric.
    metric: Met,
    /// Contains the zero layer.
    zero: Vec<NeighborNodes<M0>>,
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

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0>
where
    R: RngCore + SeedableRng,
{
    /// Creates a new HNSW with a PRNG which is default seeded to produce deterministic behavior.
    pub fn new(metric: Met) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params: Params::new(),
        }
    }

    /// Creates a new HNSW with a default seeded PRNG and with the specified params.
    pub fn new_params(metric: Met, params: Params) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params,
        }
    }

    pub fn new_with_capacity(metric: Met, params: Params, capacity: usize) -> Self {
        Self {
            metric,
            zero: Vec::with_capacity(capacity),
            features: Vec::with_capacity(capacity),
            layers: vec![],
            prng: R::from_seed(R::Seed::default()),
            params,
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Knn for Hnsw<Met, T, R, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
{
    type Ix = usize;
    type Metric = Met;
    type Point = T;
    type KnnIter = Vec<Neighbor<Met::Unit>>;

    fn knn(&self, query: &T, num: usize) -> Self::KnnIter {
        let mut searcher = Searcher::default();
        let mut neighbors = vec![
            Neighbor {
                index: !0,
                distance: Met::Unit::zero(),
            };
            num
        ];
        let found = self
            .nearest(query, num + 16, &mut searcher, &mut neighbors)
            .len();
        neighbors.resize_with(found, || unreachable!());
        neighbors
    }
}

impl<Met, T, R, const M: usize, const M0: usize> KnnPoints for Hnsw<Met, T, R, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
{
    fn get_point(&self, index: usize) -> &'_ T {
        &self.features[index]
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
{
    /// Creates a HNSW with the passed `prng`.
    pub fn new_prng(metric: Met, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params: Default::default(),
        }
    }

    /// Creates a HNSW with the passed `params` and `prng`.
    pub fn new_params_and_prng(metric: Met, params: Params, prng: R) -> Self {
        Self {
            metric,
            zero: vec![],
            features: vec![],
            layers: vec![],
            prng,
            params,
        }
    }

    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher<Met::Unit>) -> usize {
        // Get the level of this feature.
        let level = self.random_level();
        let mut cap = if level >= self.layers.len() {
            self.params.ef_construction
        } else {
            1
        };

        // If this is empty, none of this will work, so just add it manually.
        if self.is_empty() {
            // Add the zero node unconditionally.
            self.zero.push(NeighborNodes {
                neighbors: [!0; M0],
            });
            self.features.push(q);

            // Add all the layers its in.
            while self.layers.len() < level {
                // It's always index 0 with no neighbors since its the first feature.
                let node = Node {
                    zero_node: 0,
                    next_node: 0,
                    neighbors: NeighborNodes { neighbors: [!0; M] },
                };
                self.layers.push(vec![node]);
            }
            return 0;
        }

        self.initialize_searcher(&q, searcher);

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.layers.len()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher);
            cap = if ix == level {
                self.params.ef_construction
            } else {
                1
            };
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.layers.len())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(&q, searcher, Layer::NonZero(&self.layers[ix]), cap);
            // Then use the results of that search on this layer to connect the nodes.
            self.create_node(&q, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.lower_search(&self.layers[ix], searcher);
            cap = self.params.ef_construction;
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&q, searcher, cap);
        self.create_node(&q, &searcher.nearest, 0);
        // Add the feature to the zero layer.
        self.features.push(q);

        // Add all level vectors needed to be able to add this level.
        let zero_node = self.zero.len() - 1;
        while self.layers.len() < level {
            let node = Node {
                zero_node,
                next_node: self.layers.last().map(|l| l.len() - 1).unwrap_or(zero_node),
                neighbors: NeighborNodes { neighbors: [!0; M] },
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
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        self.search_layer(q, ef, 0, searcher, dest)
    }

    /// Extract the feature for a given item returned by [`HNSW::nearest`].
    ///
    /// The `item` must be retrieved from [`HNSW::search_layer`].
    pub fn feature(&self, item: usize) -> &T {
        &self.features[item as usize]
    }

    /// Extract the feature from a particular level for a given item returned by [`HNSW::search_layer`].
    pub fn layer_feature(&self, level: usize, item: usize) -> &T {
        &self.features[self.layer_item_id(level, item) as usize]
    }

    /// Retrieve the item ID for a given layer item returned by [`HNSW::search_layer`].
    pub fn layer_item_id(&self, level: usize, item: usize) -> usize {
        if level == 0 {
            item
        } else {
            self.layers[level][item as usize].zero_node
        }
    }

    pub fn layers(&self) -> usize {
        self.layers.len() + 1
    }

    pub fn len(&self) -> usize {
        self.zero.len()
    }

    pub fn layer_len(&self, level: usize) -> usize {
        if level == 0 {
            self.features.len()
        } else if level < self.layers() {
            self.layers[level - 1].len()
        } else {
            0
        }
    }

    pub fn is_empty(&self) -> bool {
        self.zero.is_empty()
    }

    pub fn layer_is_empty(&self, level: usize) -> bool {
        self.layer_len(level) == 0
    }

    /// Performs the same algorithm as [`HNSW::nearest`], but stops on a particular layer of the network
    /// and returns the unique index on that layer rather than the item index.
    ///
    /// If this is passed a `level` of `0`, then this has the exact same functionality as [`HNSW::nearest`]
    /// since the unique indices at layer `0` are the item indices.
    pub fn search_layer<'a>(
        &self,
        q: &T,
        ef: usize,
        level: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        // If there is nothing in here, then just return nothing.
        if self.features.is_empty() || level >= self.layers() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher);
        let cap = 1;

        for (ix, layer) in self.layers.iter().enumerate().rev() {
            self.search_single_layer(q, searcher, Layer::NonZero(layer), cap);
            if ix + 1 == level {
                let found = core::cmp::min(dest.len(), searcher.nearest.len());
                dest.copy_from_slice(&searcher.nearest[..found]);
                return &mut dest[..found];
            }
            self.lower_search(layer, searcher);
        }

        let cap = ef;

        // search the zero layer
        self.search_zero_layer(q, searcher, cap);

        let found = core::cmp::min(dest.len(), searcher.nearest.len());
        dest.copy_from_slice(&searcher.nearest[..found]);
        &mut dest[..found]
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_single_layer(
        &self,
        q: &T,
        searcher: &mut Searcher<Met::Unit>,
        layer: Layer<&[Node<M>]>,
        cap: usize,
    ) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            for neighbor in match layer {
                Layer::NonZero(layer) => layer[index as usize].get_neighbors(),
                Layer::Zero => self.zero[index as usize].get_neighbors(),
            } {
                let node_to_visit = match layer {
                    Layer::NonZero(layer) => layer[neighbor as usize].zero_node,
                    Layer::Zero => neighbor,
                };

                // Don't visit previously visited things. We use the zero node to allow reusing the seen filter
                // across all layers since zero nodes are consistent among all layers.
                // TODO: Use Cuckoo Filter or Bloom Filter to speed this up/take less memory.
                if searcher.seen.insert(node_to_visit) {
                    // Compute the distance of this neighbor.
                    let distance = self
                        .metric
                        .distance(q, &self.features[node_to_visit as usize]);
                    // Attempt to insert into nearest queue.
                    let pos = searcher.nearest.partition_point(|n| n.distance <= distance);
                    if pos != cap {
                        // It was successful. Now we need to know if its full.
                        if searcher.nearest.len() == cap {
                            // In this case remove the worst item.
                            searcher.nearest.pop();
                        }
                        // Either way, add the new item.
                        let candidate = Neighbor {
                            index: neighbor as usize,
                            distance,
                        };
                        searcher.nearest.insert(pos, candidate);
                        searcher.candidates.push(candidate);
                    }
                }
            }
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&self, q: &T, searcher: &mut Searcher<Met::Unit>, cap: usize) {
        self.search_single_layer(q, searcher, Layer::Zero, cap);
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn lower_search(&self, layer: &[Node<M>], searcher: &mut Searcher<Met::Unit>) {
        // Clear the candidates so we can fill them with the best nodes in the last layer.
        searcher.candidates.clear();
        // Only preserve the best candidate. The original paper's algorithm uses `1` every time.
        // See Algorithm 5 line 5 of the paper. The paper makes no further comment on why `1` was chosen.
        let &Neighbor { index, distance } = searcher.nearest.first().unwrap();
        searcher.nearest.clear();
        searcher.seen.clear();
        // Update the node to the next layer.
        let new_index = layer[index].next_node as usize;
        let candidate = Neighbor {
            index: new_index,
            distance,
        };
        searcher.seen.insert(layer[index].zero_node);
        // Insert the index of the nearest neighbor into the nearest pool for the next layer.
        searcher.nearest.push(candidate);
        // Insert the index into the candidate pool as well.
        searcher.candidates.push(candidate);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&self, q: &T, searcher: &mut Searcher<Met::Unit>) {
        // Clear the searcher.
        searcher.clear();
        // Add the entry point.
        let entry_distance = self.metric.distance(q, self.entry_feature());
        let candidate = Neighbor {
            index: 0,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
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
        let uniform: f64 = self.prng.next_u64() as f64 / core::u64::MAX as f64;
        (-libm::log(uniform) * libm::log(M as f64).recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn create_node(&mut self, q: &T, nearest: &[Neighbor<Met::Unit>], layer: usize) {
        if layer == 0 {
            let new_index = self.zero.len();
            let mut neighbors: [usize; M0] = [!0; M0];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index as usize;
            }
            let node = NeighborNodes { neighbors };
            for neighbor in node.get_neighbors() {
                self.add_neighbor(q, new_index as usize, neighbor, layer);
            }
            self.zero.push(node);
        } else {
            let new_index = self.layers[layer - 1].len();
            let mut neighbors: [usize; M] = [!0; M];
            for (d, s) in neighbors.iter_mut().zip(nearest.iter()) {
                *d = s.index;
            }
            let node = Node {
                zero_node: self.zero.len(),
                next_node: if layer == 1 {
                    self.zero.len()
                } else {
                    self.layers[layer - 2].len()
                },
                neighbors: NeighborNodes { neighbors },
            };
            for neighbor in node.get_neighbors() {
                self.add_neighbor(q, new_index, neighbor, layer);
            }
            self.layers[layer - 1].push(node);
        }
    }

    /// Attempts to add a neighbor to a target node.
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target_ix: usize, layer: usize) {
        // Get the feature for the target and get the neighbor slice for the target.
        // This is different for the zero layer.
        let (target_feature, target_neighbors) = if layer == 0 {
            (
                &self.features[target_ix],
                &self.zero[target_ix].neighbors[..],
            )
        } else {
            let target = &self.layers[layer - 1][target_ix];
            (
                &self.features[target.zero_node],
                &target.neighbors.neighbors[..],
            )
        };

        // Check if there is a point where the target has empty neighbor slots and add it there in that case.
        let empty_point = target_neighbors.partition_point(|&n| n != !0);
        if empty_point != target_neighbors.len() {
            // In this case we did find the first spot where the target was empty within the slice.
            // Now we add the neighbor to this slot.
            if layer == 0 {
                self.zero[target_ix as usize].neighbors[empty_point] = node_ix;
            } else {
                self.layers[layer - 1][target_ix as usize]
                    .neighbors
                    .neighbors[empty_point] = node_ix;
            }
        } else {
            // Otherwise, we need to find the worst neighbor currently.
            let (worst_ix, worst_distance) = target_neighbors
                .iter()
                .enumerate()
                .filter_map(|(ix, &n)| {
                    // Compute the distance to be higher than possible if the neighbor is not filled yet so its always filled.
                    if n == !0 {
                        None
                    } else {
                        // Compute the distance. The feature is looked up differently for the zero layer.
                        let distance = self.metric.distance(
                            target_feature,
                            &self.features[if layer == 0 {
                                n
                            } else {
                                self.layers[layer - 1][n].zero_node
                            }],
                        );
                        Some((ix, distance))
                    }
                })
                // This was done instead of max_by_key because min_by_key takes the first equally bad element.
                .min_by_key(|&(_, distance)| core::cmp::Reverse(distance))
                .unwrap();

            // If this is better than the worst, insert it in the worst's place.
            // This is also different for the zero layer.
            if self.metric.distance(q, target_feature) < worst_distance {
                if layer == 0 {
                    self.zero[target_ix as usize].neighbors[worst_ix] = node_ix;
                } else {
                    self.layers[layer - 1][target_ix as usize]
                        .neighbors
                        .neighbors[worst_ix] = node_ix;
                }
            }
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Default for Hnsw<Met, T, R, M, M0>
where
    R: RngCore + SeedableRng,
    Met: Default,
{
    fn default() -> Self {
        Self::new(Met::default())
    }
}
