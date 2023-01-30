use std::cmp::Ordering;

use crate::hnsw::nodes::{NeighborNodes, Node};
use crate::*;
use alloc::vec;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use space::{Metric, Neighbor};

/// This provides a HNSW implementation for any distance function.
///
/// The type `T` must implement [`space::Metric`] to get implementations.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "Met: Serialize, T: Serialize, R: Serialize",
        deserialize = "Met: Deserialize<'de>, T: Deserialize<'de>, R: Deserialize<'de>"
    ))
)]
pub struct Hnsw<Met, T, R, const M: usize, const M0: usize>
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
{
    metric: Met,
    prng: R,
    storage: NodeDB<T, M, M0>,
    params: Params,
}

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0>
where
    R: RngCore + SeedableRng,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
{
    pub fn new(metric: Met) -> Self {
        let params = Params::default();
        let prng = R::from_seed(R::Seed::default());
        let storage = NodeDB::<T, M, M0>::new(&params.dbpath);
        Self { metric, prng, storage, params }
    }
    pub fn new_with_params(metric: Met, params: Params) -> Self {
        let prng = R::from_seed(R::Seed::default());
        let storage = NodeDB::<T, M, M0>::new(&params.dbpath);
        Self { metric, prng, storage, params }
    }
    pub fn new_with_prng(metric: Met, prng: R) -> Self {
        let params = Params::default();
        let storage = NodeDB::<T, M, M0>::new(&params.dbpath);
        Self { metric, prng, storage, params }
    }
    pub fn new_with_params_and_prng(metric: Met, params: Params, prng: R) -> Self {
        let storage = NodeDB::<T, M, M0>::new(&params.dbpath);
        Self { metric, prng, storage, params }
    }
}
/*
impl<Met, T, R, const M: usize, const M0: usize> Knn for Hnsw<Met, T, R, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
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
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
{
    fn get_point(&self, index: usize) -> &'_ T {
        //TODO: error handling
        self.storage.get(index as i32).unwrap().unwrap().feature
    }
}
*/

impl<Met, T, R, const M: usize, const M0: usize> Hnsw<Met, T, R, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
{
    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, searcher: &mut Searcher<Met::Unit>) -> usize {
        // Get the level of this feature.
        let level = self.random_level();
        let mut cap = if level >= self.storage.num_layer() {
            self.params.ef_construction
        } else {
            1
        };
        // instantiate new node
        let mut node = Node::<T, M, M0> {
            id: 0,
            neighbors: vec![],
            zero_neighbors: NeighborNodes::new(),
            feature: q,
        };

        // If this is empty, none of this will work, so just add it manually.
        if self.storage.num_node() == 0 {
            // Add all the layers its in.
            for _i in 0..level {
                // It's always index 0 with no neighbors since its the first feature.
                node.neighbors.push(NeighborNodes::new());
            }
            
            let _ = self.storage.update_entry_point(node.id as i32);
            let _ = self.storage.put(node);

            return 0;
        }

        node.id = self.storage.meta_data.num_nodes.unwrap() as usize;
        self.initialize_searcher(&node.feature, searcher);

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.storage.num_layer()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(ix, &node.feature, searcher, cap);
            // Then lower the search only after we create the node.
            self.go_down_layer(searcher);
            cap = if ix == level {
                self.params.ef_construction
            } else {
                1
            };
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.storage.num_layer())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(ix, &node.feature, searcher, cap);
            // Then use the results of that search on this layer to connect the nodes.
            self.update_neighbors(&node, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.go_down_layer(searcher);
            cap = self.params.ef_construction;
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&node.feature, searcher, cap);
        self.update_neighbors(&node, &searcher.nearest, 0);

        self.storage.num_node() - 1
    }

    /// Does a k-NN search where `q` is the query element and it attempts to put up to `M` nearest neighbors into `dest`.
    /// `ef` is the candidate pool size. `ef` can be increased to get better recall at the expense of speed.
    /// If `ef` is less than `dest.len()` then `dest` will only be filled with `ef` elements.
    ///
    /// Returns a slice of the filled neighbors.
    pub fn nearest<'a>(
        &mut self,
        q: &T,
        ef: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        self.search_layer(q, ef, 0, searcher, dest)
    }
/* 
    pub fn is_empty(&self) -> bool {
        if let Ok(Some(n)) = self.storage.get(0) {
            return false;
        }
        return true;
    }
*/
    /// Performs the same algorithm as [`HNSW::nearest`], but stops on a particular layer of the network
    /// and returns the unique index on that layer rather than the item index.
    ///
    /// If this is passed a `level` of `0`, then this has the exact same functionality as [`HNSW::nearest`]
    /// since the unique indices at layer `0` are the item indices.
    pub fn search_layer<'a>(
        &mut self,
        q: &T,
        ef: usize,
        level: usize,
        searcher: &mut Searcher<Met::Unit>,
        dest: &'a mut [Neighbor<Met::Unit>],
    ) -> &'a mut [Neighbor<Met::Unit>] {
        // If there is nothing in here, then just return nothing.
        if self.storage.num_layer() == 0 || level >= self.storage.num_layer() {
            return &mut [];
        }

        self.initialize_searcher(q, searcher);
        let cap = 1;

        for idx in (0..self.storage.num_layer()).rev() {
            self.search_single_layer(idx, q, searcher, cap);
            if idx + 1 == level {
                let found = core::cmp::min(dest.len(), searcher.nearest.len());
                dest.copy_from_slice(&searcher.nearest[..found]);
                return &mut dest[..found];
            }
            self.go_down_layer(searcher);
        }

        let cap = ef;

        self.search_zero_layer(q, searcher, cap);

        let found = core::cmp::min(dest.len(), searcher.nearest.len());
        dest.copy_from_slice(&searcher.nearest[..found]);
        &mut dest[..found]
    }

    /// Greedily finds the approximate nearest neighbors to `q` in a non-zero layer.
    /// This corresponds to Algorithm 2 in the paper.
    fn search_single_layer(&mut self, layer_idx: usize, q: &T, searcher: &mut Searcher<Met::Unit>, cap: usize) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            let candidate_node = self
                .storage
                .get(index as i32)
                .expect(format!("unknown node id: {}", index).as_str())
                .expect(format!("unknown node id: {}", index).as_str());

            candidate_node
                .neighbors[layer_idx]
                .neighbors()
                .for_each(|nidx| {
                    let neighbor_node = self
                        .storage
                        .get(nidx.0 as i32)
                        .expect(format!("unknown node id: {}", index).as_str())
                        .expect(format!("unknown node id: {}", index).as_str());
                    if searcher.seen.insert(nidx.0) {
                        let d = self.metric.distance(q, &neighbor_node.feature);
                        let pos = searcher.nearest.partition_point(|n| n.distance <= d);
                        if pos != cap {
                            if searcher.nearest.len() == cap {
                                searcher.nearest.pop();
                            }
                            let new_candidate = Neighbor {
                                index: nidx.0,
                                distance: d,
                            };
                            searcher.nearest.insert(pos, new_candidate);
                            searcher.candidates.push(new_candidate);
                        }
                    }
                });
        }
    }

    /// Greedily finds the approximate nearest neighbors to `q` in the zero layer.
    fn search_zero_layer(&mut self, q: &T, searcher: &mut Searcher<Met::Unit>, _cap: usize) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            let candidate_node = self
                .storage
                .get(index as i32)
                .expect(format!("unknown node id: {}", index).as_str())
                .expect(format!("unknown node id: {}", index).as_str());

            candidate_node
                .zero_neighbors
                .neighbors()
                .for_each(|nidx| {
                    let neighbor_node = self
                        .storage
                        .get(nidx.0 as i32)
                        .expect(format!("unknown node id: {}", index).as_str())
                        .expect(format!("unknown node id: {}", index).as_str());
                    if searcher.seen.insert(nidx.0) {
                        let d = self.metric.distance(q, &neighbor_node.feature);
                        let pos = searcher.nearest.partition_point(|n| n.distance <= d);
                        let new_candidate = Neighbor {
                            index: nidx.0,
                            distance: d,
                        };
                        searcher.nearest.insert(pos, new_candidate);
                        searcher.candidates.push(new_candidate);
                    }
                });
        }
    }

    /// Ready a search for the next level down.
    ///
    /// `m` is the maximum number of nearest neighbors to consider during the search.
    fn go_down_layer(&self, searcher: &mut Searcher<Met::Unit>) {
        searcher.candidates.clear();
        let next = searcher.nearest.first().unwrap().clone();
        searcher.nearest.clear();

        searcher.nearest.push(next.clone());
        searcher.candidates.push(next);
    }

    /// Resets a searcher, but does not set the `cap` on the nearest neighbors.
    /// Must be passed the query element `q`.
    fn initialize_searcher(&mut self, q: &T, searcher: &mut Searcher<Met::Unit>) {
        let ep = self.storage.load_entry_point_node().unwrap();
        searcher.clear();
        // Add the entry point.
        let entry_distance = self.metric.distance(q, &ep.feature);
        let candidate = Neighbor {
            index: self.storage.meta_data.entry_point.unwrap() as usize,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(0);
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.next_u64() as f64 / core::u64::MAX as f64;
        (-libm::log(uniform) * libm::log(M as f64).recip()) as usize
    }

    /// Creates a new node at a layer given its nearest neighbors in that layer.
    /// This contains Algorithm 3 from the paper, but also includes some additional logic.
    fn update_neighbors(&mut self, node: &Node<T, M, M0>, nearest: &[Neighbor<Met::Unit>], layer: usize) {
        nearest.iter()
               .for_each(|id| {
                    if let Ok(Some(mut n)) = self.storage.get(id.index as i32) {
                        self.add_neighbor(&node.feature, node.id, &mut n, layer);
                        let _ = self.storage.put(n);
                    }
               });
    }

    /// Attempts to add a neighbor to a target node.
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target : &mut Node<T, M, M0>, layer: usize) {
        let distance :f32 = self.metric.distance(q, &target.feature).into();
        let insert_point = target.neighbors[layer]
                                        .neighbors
                                        .partition_point(|&n| n.1 < distance);
        let length = if layer == 0 { M0 } else { M };
        let mut buf = (0, f32::MAX);
        for i in 0..length {
            target.neighbors[layer].neighbors[i] = match i.cmp(&insert_point) {
                Ordering::Equal => {
                    (node_ix, distance)
                },
                Ordering::Greater => {
                    buf
                },
                Ordering::Less => {
                    target.neighbors[layer].neighbors[i]
                }
            };
            buf = target.neighbors[layer].neighbors[i];
        }
    }
}

impl<Met, T, R, const M: usize, const M0: usize> Default for Hnsw<Met, T, R, M, M0>
where
    R: RngCore + SeedableRng,
    Met: Metric<T> + Default,
    <Met as Metric<T>>::Unit: Into<f32> + From<f32>,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone,
{
    fn default() -> Self {
        Self::new(Met::default())
    }
}
