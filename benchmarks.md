# Benchmarks

The above benchmark shows the speed boost this gives for 10-NN, but this datastructure is intended (by me) to be used for fast feature matching in images. When matching features in images, they must satisfy the lowe's ratio, which means that the first match must be significantly better than the second match or there may be ambiguity. For that purpose, here are several 2-NN benchmarks for 256-bit binary descriptors:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_10000_m12.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_100000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn2_256bit_1000000_m24.svg)

Please note that `M = 24` is not optimal for the above graph at 10000 items. The default `M` on this implementation is `12`, which can be fairly optimal for the 2-NN case.

You might want to know the performance of HNSW and how it changes when you increase the number of nearest neighbors. Here are some graphs of some 50-NN searches with different `M`:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m12.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn50_256bit_10000_m52.svg)

Finally, here are some more 10-NN benchmarks to satisfy your curiosity:

![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_10000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_100000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_1000000_m24.svg)
![Recall Graph](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/nn10_256bit_1000000_m48.svg)

You can find benchmarks of HNSW with 256-bit binary features vs linear search on 2-NN with `ef` parameter of `24`, `M` parameter of `24` (very high recall), and `efConstruction` set to `400` [here](http://vadixidav.github.io/hnsw/0949a5a503402a8f0effef01b42b5360c83c688a/report/index.html). This compares it against linear search, which is pretty fast in small datasets. This is not really a valid comparison because it is comparing apples to oranges (a linear search which is perfect with an ANN algorithm that is getting worse at recall). However, this benchmark is useful for profiling the code, so I will share its results here. Please use the recall graphs above as the actual point of comparison.

You can also generate recall graphs (native cpu instructions recommended). Use the following to see how:

```bash
# Discrete version (binary descriptors)
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example recall_discrete -- --help
# Non-discrete version (floating descriptors)
RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --example recall -- --help
```

If you don't want to use random data (highly recommended), please use the `generate.py` script like so:

```bash
# AKAZE binary descriptors
python3 scripts/generate.py akaze 2011_09_29/2011_09_29_drive_0071_extract/image_00/data/*.png > data/akaze
# KAZE floating point descriptors
python3 scripts/generate.py kaze 2011_09_29/2011_09_29_drive_0071_extract/image_00/data/*.png > data/kaze
```

This code ran againt the Kitti dataset ` 2011_09_29_drive_0071`. You can find that [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). Download "[unsynced+unrectified data]". It is a large (4.1 GB) download, so it may take some time.

You can still run the above generation against any dataset you would like if you would like to test its performance on said dataset.

This crate may take a while to compile due to the use of `typenum` and `generic-array`. If you dislike this, consider contributing to some issues labeled [A-const-generics](https://github.com/rust-lang/rust/labels/A-const-generics) in Rust to help push along the const generics support in the compiler. The `recall_discrete` generator is especially time-consuming to build.
