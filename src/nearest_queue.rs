use std::fmt;

/// This keeps the nearest `cap` items at all times.
///
/// It is efficiently implemented for nearest neighbor searches to have constant-time insertion, but
/// it only works for distances in the range [0, 128]. This is specifically tailored for doing hamming space
/// nearest neighbor searches.
#[derive(Clone)]
pub struct NearestQueue<T> {
    cap: usize,
    size: usize,
    worst: u32,
    distances: [Vec<T>; 129],
}

impl<T> NearestQueue<T> {
    /// This sets the capacity of the queue to `cap`, meaning that adding items to the queue will eject the worst ones
    /// if they are better once `cap` is reached. If the capacity is lowered, this removes the worst elements to
    /// keep `size == cap`.
    pub(crate) fn set_capacity(&mut self, cap: usize) {
        self.set_size(cap);
        self.cap = cap;
    }

    /// This removes elements until it reaches `size`. If `size` is lower than the current
    /// number of elements, this does nothing. If the size is lowered, this will unconditionally allow insertions
    /// until `cap` is reached.
    pub(crate) fn set_size(&mut self, size: usize) {
        if size == 0 {
            for v in &mut self.distances[..] {
                v.clear();
            }
            self.size = 0;
            self.worst = 128;
        } else if size < self.size {
            // Remove the difference between them.
            for _ in size..self.size {
                self.remove_worst();
            }
            self.size = size;
            self.worst = 128;
        }
    }

    /// Reset the queue while maintaining the allocated memory.
    pub(crate) fn reset(&mut self, cap: usize) {
        assert_ne!(cap, 0);
        self.cap = cap;
        self.size = 0;
        self.worst = 128;
        for v in self.distances.iter_mut() {
            v.clear();
        }
    }

    /// Add a feature to the search.
    pub(crate) fn insert(&mut self, item: T, distance: u32) -> bool {
        if self.size != self.cap {
            self.distances[distance as usize].push(item);
            self.size += 1;
            // Set the worst feature appropriately.
            if self.size == self.cap {
                self.update_worst();
            }
            true
        } else if distance < self.worst {
            self.distances[distance as usize].push(item);
            self.remove_worst();
            true
        } else {
            false
        }
    }

    /// Gets the worst distance in the queue currently.
    ///
    /// This is initialized to 128 (which is the worst possible distance) until `cap` elements have been inserted.
    pub(crate) fn worst(&self) -> u32 {
        self.worst
    }

    /// Add a feature to the search with the precondition we are already at the cap.
    fn add_one_cap(&mut self, item: T, distance: u32) {
        // We stop searching once we have enough features under the search distance,
        // so if this is true it will always get added to the FeatureHeap.
        if distance < self.worst {
            self.distances[distance as usize].push(item);
            self.remove_worst();
        }
    }

    /// Updates the worst when it has been set.
    fn update_worst(&mut self) {
        self.worst -= self.distances[0..=self.worst as usize]
            .iter()
            .rev()
            .position(|v| !v.is_empty())
            .unwrap() as u32;
    }

    /// Remove the worst item and update the worst distance.
    fn remove_worst(&mut self) {
        self.distances[self.worst as usize].pop();
        self.update_worst();
    }

    /// Fill a slice with the `top` elements and return the part of the slice written.
    pub fn fill_slice<'a>(&self, s: &'a mut [T]) -> &'a mut [T]
    where
        T: Clone,
    {
        let total_fill = std::cmp::min(s.len(), self.size);
        for (ix, f) in self
            .distances
            .iter()
            .flat_map(|v| v.iter())
            .take(total_fill)
            .enumerate()
        {
            s[ix] = f.clone();
        }
        &mut s[0..total_fill]
    }

    /// Iterate over the entire queue in best-to-worse order.
    pub(crate) fn iter<'a>(&'a mut self) -> impl Iterator<Item = (&'a T, u32)> {
        self.distances
            .iter()
            .enumerate()
            .flat_map(|(distance, v)| v.iter().map(move |item| (item, distance as u32)))
    }
    /// Iterate over the entire queue in best-to-worse order.
    pub(crate) fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a mut T, u32)> {
        self.distances
            .iter_mut()
            .enumerate()
            .flat_map(|(distance, v)| v.iter_mut().map(move |item| (item, distance as u32)))
    }
}

impl<T> fmt::Debug for NearestQueue<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.distances[..].fmt(formatter)
    }
}

impl<T> Default for NearestQueue<T> {
    fn default() -> Self {
        Self {
            cap: 0,
            size: 0,
            worst: 128,
            distances: [
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ],
        }
    }
}
