mod feature_heap;
mod hamming_queue;

const M: usize = 12;
const M_MAX: usize = M;
const M_MAX0: usize = M * 2;
// Can't do this because `ln` and `recip` are not const functions yet.
// const ML: f32 = (M as f32).ln().recip();
const ML: f32 = 0.40242960438;

/// This provides a HNSW implementation for 128-bit hamming space.
pub struct HNSW {
    /// Contains the zero layer.
    zero: Vec<ZeroNode>,
    /// Contains each non-zero layer.
    layers: Vec<Vec<Node>>,
    /// The features which are referenced by the NSWs.
    /// This is the index into the highest-level world currently existing that we will start searching from.
    enter_index: u32,
}

/// A node in the zero layer.
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

/// A node in any other layer.
struct Node {
    /// The node in the zero layer this refers to.
    zero_node: u32,
    neighbors: [u32; M_MAX],
}
