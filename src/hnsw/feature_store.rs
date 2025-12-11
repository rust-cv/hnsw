use alloc::vec::Vec;

/// Trait for abstracting feature storage in HNSW.
pub trait FeatureStore<T> {
    fn get(&self, index: usize) -> &T;
    fn push(&mut self, feature: T);
    fn len(&self) -> usize;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> FeatureStore<T> for Vec<T> {
    #[inline]
    fn get(&self, index: usize) -> &T {
        &self[index]
    }

    #[inline]
    fn push(&mut self, feature: T) {
        Vec::push(self, feature)
    }

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        Vec::is_empty(self)
    }
}
