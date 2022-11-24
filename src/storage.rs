extern crate tempdir;
extern crate leveldb;

use std::num::NonZeroUsize;

use leveldb::database::Database;
use leveldb::kv::KV;
use leveldb::options::{Options, WriteOptions, ReadOptions};

use lru::LruCache;

use rmps::{Serializer, Deserializer};

use crate::hnsw::nodes::Node;

enum Error {
    Serialize(Serializer::Error),
    Deserialize(Deserializer::Error),
    LevelDB(leveldb::error::Error),
    UnknownError,
}

impl From<Serializer::Error> for Error {
    fn from(e: Serializer::Error) -> Self {
        Self::Serialize(e)
    }
}

impl From<Deserializer::Error> for Error {
    fn from(e: Deserializer::Error) -> Self {
        Self::Deserialize(e)
    }
}

impl From<leveldb::error::Error> for Error {
    fn from(e: leveldb::error::Error) -> Self {
        Self::LevelDB(e)
    }
}

pub struct NodeDB<NodeType> {
    db: Database<usize>,
    cache: LruCache<usize, NodeType>,
}

impl<NodeType> NodeDB<NodeType> {
    fn new(path: &String) -> Self {
        db = Database::open::<usize>::(path, {
            let mut opts = leveldb::Options::new();
            opts.create_if_missing = true;
            
            opts
        });
        cache = LruCache::new(NonZeroUsize::new(10000).unwrap());

        NodeDB<NodeType> { db, cache }
    }
    fn put(node: NodeType) -> Result<Error> {
        let mut buf = Vec::new();
        node.serialize(Serializer::new(&buf))?;

        db.put(
            WriteOptions::new(),
            node.id,
            buf.as_bytes(),
        );
        cache.insert(node.id, node);
    }
    fn get(id: usize) -> Result<&'_ NodeType, Error> {
        if let Some(&mut val) = cache.get_mut(id) {
            return Ok(&val);
        }
        let data = db.get(
            ReadOptions::new(),
            id,
        )?;
        let node: NodeType = rmp_serde::from_slice(&data[..])?;
        cache.insert(id, node);
        if let Some(&mut val) = cache.get_mut(id) {
            return Ok(&val);
        }
        return Err(Error::UnknownError);
    }
}

pub struct MetadataDB {
    db: Database<String>,
    cache: LruCache<usize, NodeType>,
}