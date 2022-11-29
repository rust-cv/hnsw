use leveldb::{
    database::Database,
    kv::KV,
    options::{Options, ReadOptions, WriteOptions},
};
use lru::LruCache;
use rmp_serde::Serializer;
use serde::Serialize as _;
use std::num::NonZeroUsize;

#[derive(Debug)]
enum Error {
    Serialize(rmp_serde::encode::Error),
    Deserialize(rmp_serde::decode::Error),
    LevelDB(leveldb::error::Error),
    UnknownError,
}

impl From<rmp_serde::encode::Error> for Error {
    fn from(e: rmp_serde::encode::Error) -> Self {
        Self::Serialize(e)
    }
}

impl From<rmp_serde::decode::Error> for Error {
    fn from(e: rmp_serde::decode::Error) -> Self {
        Self::Deserialize(e)
    }
}

impl From<leveldb::error::Error> for Error {
    fn from(e: leveldb::error::Error) -> Self {
        Self::LevelDB(e)
    }
}

pub struct NodeDB<T, const M: usize, const M0: usize>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    db: Database<i32>,
    cache: LruCache<i32, crate::nodes::Node<T, M, M0>>,
}

impl<T, const M: usize, const M0: usize> NodeDB<T, M, M0>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        let db = Database::open(path.as_ref(), {
            let mut opts = Options::new();
            opts.create_if_missing = true;

            opts
        })
        .unwrap();

        let cache = LruCache::new(NonZeroUsize::new(10000).unwrap());

        NodeDB { db, cache }
    }

    pub fn put(&self, node: crate::nodes::Node<T, M, M0>) -> Result<(), Error> {
        let mut buf = Vec::new();
        node.serialize(&mut Serializer::new(&mut buf))?;

        self.db.put(WriteOptions::new(), node.id as i32, &buf);
        self.cache.put(node.id as i32, node);

        Ok(())
    }

    pub fn get(&self, id: i32) -> Result<Option<&'_ crate::nodes::Node<T, M, M0>>, Error> {
        if let Some(&mut val) = self.cache.get_mut(&id) {
            return Ok(Some(&val));
        }

        if let Some(data) = self.db.get(ReadOptions::new(), id)? {
            let node = rmp_serde::from_slice(&data[..])?;
            self.cache.put(id, node);

            Ok(Some(&node))
        } else {
            Ok(None)
        }
    }
}

pub struct MetadataDB<T, const M: usize, const M0: usize>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    db: Database<i32>,
    cache: LruCache<usize, crate::hnsw::nodes::Node<T, M, M0>>,
}
