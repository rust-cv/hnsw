use std::cmp::Ordering;

use crate::{
    metric::{Metric, Neighbor},
    nodes::{NeighborNodes, Node},
    storage::Storage,
    HashSet, Params, RandomState,
};
use alloc::vec;
use rand::Rng;
use rand_core::{RngCore, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
struct Searcher<Metric> {
    candidates: Vec<Neighbor<Metric>>,
    nearest: Vec<Neighbor<Metric>>,
    seen: HashSet<usize, RandomState>,
}

impl<Metric> Searcher<Metric> {
    pub fn new() -> Self {
        Self {
            candidates: vec![],
            nearest: vec![],
            seen: HashSet::with_hasher(RandomState::with_seeds(0, 0, 0, 0)),
        }
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

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
pub struct Hnsw<Met, T, R, S, const M: usize, const M0: usize>
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone + core::fmt::Debug,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<u32> + From<u32> + core::fmt::Debug,
    S: Storage<T, M, M0>,
{
    metric: Met,
    prng: R,
    storage: S,
    params: Params,
    _phantom: std::marker::PhantomData<T>,
}

impl<Met, T, R, S, const M: usize, const M0: usize> Hnsw<Met, T, R, S, M, M0>
where
    R: RngCore + SeedableRng,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<u32> + From<u32> + core::fmt::Debug,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone + core::fmt::Debug,
    S: Storage<T, M, M0>,
{
    pub fn new(metric: Met, storage: S) -> Self {
        Self {
            metric,
            prng: R::from_entropy(),
            storage,
            params: Params::default(),
            _phantom: std::marker::PhantomData {},
        }
    }

    pub fn new_with_params(metric: Met, storage: S, params: Params) -> Self {
        Self {
            metric,
            prng: R::from_entropy(),
            storage,
            params,
            _phantom: std::marker::PhantomData {},
        }
    }

    pub fn new_with_prng(metric: Met, prng: R, storage: S) -> Self {
        Self {
            metric,
            prng,
            storage,
            params: Params::default(),
            _phantom: std::marker::PhantomData {},
        }
    }

    pub fn new_with_params_and_prng(metric: Met, params: Params, prng: R, storage: S) -> Self {
        Self {
            metric,
            prng,
            storage,
            params,
            _phantom: std::marker::PhantomData {},
        }
    }
}

impl<Met, T, R, S, const M: usize, const M0: usize> Hnsw<Met, T, R, S, M, M0>
where
    R: RngCore,
    Met: Metric<T>,
    <Met as Metric<T>>::Unit: Into<u32> + From<u32> + core::fmt::Debug,
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone + core::fmt::Debug,
    S: Storage<T, M, M0>,
{
    pub fn insert_with_auto_id(&mut self, q: T) -> usize {
        let id = self.storage.num_node();
        self.insert(q, id)
    }
    /// Inserts a feature into the HNSW.
    pub fn insert(&mut self, q: T, id: usize) -> usize {
        // Get the level of this feature.
        let level = self.random_level();
        let mut cap = if level >= self.storage.num_layer() {
            self.params.ef_construction
        } else {
            1
        };
        // instantiate new node
        let mut node = Node::<T, M, M0> {
            id,
            neighbors: vec![],
            zero_neighbors: NeighborNodes::new(),
            feature: q,
        };
        for _i in 0..level {
            node.neighbors.push(NeighborNodes::new());
        }

        // If this is empty, none of this will work, so just add it manually.
        if self.storage.num_node() == 0 {
            // Add all the layers its in.
            let _ = self.storage.update_entry_point(node.id as i32);
            let _ = self.storage.store_new_node(node);

            return 0;
        }

        let mut searcher = self.initialize_searcher(&node.feature);

        // Find the entry point on the level it was created by searching normally until its level.
        for ix in (level..self.storage.num_layer()).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(ix, &node.feature, &mut searcher, cap);
            // Then lower the search only after we create the node.
            self.go_down_layer(&mut searcher);
            cap = if ix == level {
                self.params.ef_construction
            } else {
                1
            };
        }

        // Then start from its level and connect it to its nearest neighbors.
        for ix in (0..core::cmp::min(level, self.storage.num_layer())).rev() {
            // Perform an ANN search on this layer like normal.
            self.search_single_layer(ix, &node.feature, &mut searcher, cap);
            // Then use the results of that search on this layer to connect the nodes.

            self.assign_neighbors::<M>(&mut node, &searcher.nearest, ix + 1);
            self.update_neighbors(&node, &searcher.nearest, ix + 1);
            // Then lower the search only after we create the node.
            self.go_down_layer(&mut searcher);
            cap = self.params.ef_construction;
        }

        // Also search and connect the node to the zero layer.
        self.search_zero_layer(&node.feature, &mut searcher, cap);
        self.update_neighbors(&node, &searcher.nearest, 0);

        if self.storage.num_layer() < node.neighbors.len() {
            let _ = self.storage.update_entry_point(node.id as i32);
        }
        let _ = self.storage.store_new_node(node);

        self.storage.num_node() - 1
    }

    pub fn nearest(&mut self, q: &T, ef: usize, take: usize) -> Vec<Neighbor<Met::Unit>> {
        self.search_layer(q, ef, 0).into_iter().take(take).collect()
    }

    pub fn search_layer(&mut self, q: &T, ef: usize, level: usize) -> Vec<Neighbor<Met::Unit>> {
        // If there is nothing in here, then just return nothing.
        if self.storage.num_node() == 0 || level > self.storage.num_layer() {
            return vec![];
        }

        let mut searcher = self.initialize_searcher(q);
        let cap = 1;
        for idx in (0..self.storage.num_layer()).rev() {
            self.search_single_layer(idx, q, &mut searcher, cap);
            if idx + 1 == level {
                return searcher.nearest;
            }
            self.go_down_layer(&mut searcher);
        }

        let cap = ef;

        self.search_zero_layer(q, &mut searcher, cap);

        searcher.nearest
    }

    fn search_single_layer(
        &mut self,
        layer_idx: usize,
        q: &T,
        searcher: &mut Searcher<Met::Unit>,
        cap: usize,
    ) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            let candidate_node = self
                .storage
                .get(index as i32)
                .expect("unknown node id")
                .expect("unknown node id");
            if candidate_node.neighbors.get(layer_idx).is_none() {
                println!("candidate {:?}", candidate_node);
                self.dump();
            }
            candidate_node.neighbors[layer_idx]
                .neighbors()
                .for_each(|nidx| {
                    let neighbor_node = self
                        .storage
                        .get(nidx.0 as i32)
                        .expect("unknown node id")
                        .expect("unknown node id");
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

    fn search_zero_layer(&mut self, q: &T, searcher: &mut Searcher<Met::Unit>, _cap: usize) {
        while let Some(Neighbor { index, .. }) = searcher.candidates.pop() {
            let candidate_node = self
                .storage
                .get(index as i32)
                .expect("unknown node id")
                .expect("unknown node id");

            candidate_node.zero_neighbors.neighbors().for_each(|nidx| {
                let neighbor_node = self
                    .storage
                    .get(nidx.0 as i32)
                    .expect("unknown node id")
                    .expect("unknown node id");
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

    fn go_down_layer(&self, searcher: &mut Searcher<Met::Unit>) {
        searcher.candidates.clear();
        let next = *searcher.nearest.first().unwrap();
        searcher.nearest.clear();

        searcher.nearest.push(next);
        searcher.candidates.push(next);
    }

    fn initialize_searcher(&mut self, q: &T) -> Searcher<Met::Unit> {
        let mut searcher = Searcher::new();
        let ep = self.storage.load_entry_point_node().unwrap();
        searcher.clear();
        // Add the entry point.
        let entry_distance = self.metric.distance(q, &ep.feature);
        let candidate = Neighbor {
            index: self.storage.meta_data().entry_point.unwrap() as usize,
            distance: entry_distance,
        };
        searcher.candidates.push(candidate);
        searcher.nearest.push(candidate);
        searcher.seen.insert(0);

        searcher
    }

    /// Generates a correctly distributed random level as per Algorithm 1 line 4 of the paper.
    fn random_level(&mut self) -> usize {
        let uniform: f64 = self.prng.gen();
        (-libm::log2(uniform) * libm::log2(M as f64).recip()) as usize
    }

    fn update_neighbors(
        &mut self,
        node: &Node<T, M, M0>,
        nearest: &[Neighbor<Met::Unit>],
        layer: usize,
    ) {
        nearest.iter().for_each(|id| {
            if let Ok(Some(mut n)) = self.storage.get(id.index as i32) {
                self.add_neighbor(&node.feature, node.id, &mut n, layer);
                let _ = self.storage.put(n);
            }
        });
    }

    fn  assign_neighbors<const MM: usize>(
        &mut self,
        node: &mut Node<T, M, M0>,
        nearest: &[Neighbor<Met::Unit>],
        layer: usize,
    ) {
        for i in 0..std::cmp::min(MM, nearest.len()) {
            let neighbor = nearest[i];
            node.neighbors[layer-1].neighbors[i] = (neighbor.index, neighbor.distance.into());
        }
    }

    fn add_neighbor_internal<const MM: usize>(
        &mut self,
        node_ix: usize,
        target: &mut NeighborNodes<MM>,
        distance: Met::Unit,
    ) {
        let insert_point = target
            .neighbors
            .partition_point(|&n| Met::Unit::from(n.1) < distance);
        let mut buf = (0, u32::MAX);
        for i in 0..MM {
            let curr = target.neighbors[i];
            target.neighbors[i] = match i.cmp(&insert_point) {
                Ordering::Equal => (node_ix, distance.into()),
                Ordering::Greater => buf,
                Ordering::Less => target.neighbors[i],
            };
            buf = curr;
        }
    }
    fn add_neighbor(&mut self, q: &T, node_ix: usize, target: &mut Node<T, M, M0>, layer: usize) {
        let distance = self.metric.distance(q, &target.feature);
        if layer == 0 {
            self.add_neighbor_internal(node_ix, &mut target.zero_neighbors, distance);
        } else {
            self.add_neighbor_internal(node_ix, &mut target.neighbors[layer - 1], distance);
        };
    }

    pub fn dump(&mut self) {
        for i in 0..self.storage.num_node() {
            println!("{:?}", self.storage.get(i as i32));
        }
        println!("{:?}", self.storage.meta_data());
    }
}
