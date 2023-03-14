//! Useful tests for debugging since they are hand-written and easy to see the debugging output.

use hnsw::{
    metric::{EncodableFloat, Neighbor, SimpleEuclidean},
    storage, Hnsw,
};
use rand_pcg::Pcg64;

fn test_hnsw() -> Hnsw<SimpleEuclidean, Vec<f32>, Pcg64, storage::HashMap<Vec<f32>, 12, 24>, 12, 24>
{
    let mut hnsw = Hnsw {
        metric: SimpleEuclidean {},
        ..Default::default()
    };

    let features = [
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 1.0],
    ];

    for feature in features {
        hnsw.insert(feature);
    }

    hnsw
}

#[test]
fn nearest_neighbor() {
    let mut hnsw = test_hnsw();
    let input = vec![0.0, 0.0, 0.0, 1.0];
    let mut neighbors: Vec<_> = hnsw.nearest(&input, 24, 8);

    // Distance 1
    neighbors[1..3].sort_unstable();
    // Distance 2
    neighbors[3..6].sort_unstable();
    // Distance 3
    neighbors[6..8].sort_unstable();

    assert_eq!(
        neighbors,
        [
            Neighbor {
                index: 0,
                distance: EncodableFloat { value: 0.0 }
            },
            Neighbor {
                index: 4,
                distance: EncodableFloat { value: 1.0 }
            },
            Neighbor {
                index: 7,
                distance: EncodableFloat { value: 1.0 }
            },
            Neighbor {
                index: 1,
                distance: EncodableFloat {
                    value: (2.0_f32).sqrt()
                }
            },
            Neighbor {
                index: 2,
                distance: EncodableFloat {
                    value: (2.0_f32).sqrt()
                }
            },
            Neighbor {
                index: 3,
                distance: EncodableFloat {
                    value: (2.0_f32).sqrt()
                }
            },
            Neighbor {
                index: 5,
                distance: EncodableFloat {
                    value: (3.0_f32).sqrt()
                }
            },
            Neighbor {
                index: 6,
                distance: EncodableFloat {
                    value: (3.0_f32).sqrt()
                }
            }
        ]
    );
}
