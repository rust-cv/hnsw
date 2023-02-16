// use smallvec::SmallVec;

/// A node in the zero layer
#[derive(Clone, Debug)]
pub struct NeighborNodes<const N: usize> {
    /// The neighbors of this node.
    pub neighbors: [(usize, u32); N], // (id, cost)
}

impl<const N: usize> NeighborNodes<N> {
    pub fn new() -> Self {
        NeighborNodes {
            neighbors: [(0, u32::MAX); N],
        }
    }
    pub fn neighbors(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.neighbors
            .iter()
            .cloned()
            .take_while(|&n| n.1 != u32::MAX)
    }
}

/// A node in any other layer other than the zero layer
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct Node<T, const M: usize, const M0: usize>
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::clone::Clone + core::fmt::Debug,
{
    pub id: usize,
    /// The node in the zero layer this refers to.
    //pub zero_node: usize,
    /// The node in the layer below this one that this node corresponds to.
    //pub next_node: usize,
    /// The neighbors in the graph of this node.
    pub neighbors: Vec<NeighborNodes<M>>,
    pub zero_neighbors: NeighborNodes<M0>,
    pub feature: T,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct MetaData {
    pub entry_point: Option<i32>,
    pub num_nodes: Option<i32>,
}

/*
impl<T, const M: usize, const M0: usize> Node<T, M, M0>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    pub fn neighbors(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors
            .iter()
            .map(|neighbors| neighbors.neighbors())
            .flatten()
    }
}
*/

/*
/// The inbound nodes that are pointing to this node.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct InboundNodes<const N: usize> {
    pub neighbors: SmallVec<[usize; N]>,
}

*/
