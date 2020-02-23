mod neighbors;

use criterion::*;

criterion_main! {
    neighbors::benches,
}
