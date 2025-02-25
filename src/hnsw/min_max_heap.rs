use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct MinMaxHeap<T: Ord> {
    elements: Vec<T>,
}

impl<T: Ord> MinMaxHeap<T> {
    pub fn new() -> Self {
        MinMaxHeap {
            elements: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        MinMaxHeap {
            elements: Vec::with_capacity(capacity),
        }
    }

    pub fn clear(&mut self) {
        self.elements.clear();
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn peek_min(&self) -> Option<&T> {
        match self.elements.len() {
            0 => None,
            1 => Some(&self.elements[0]),
            _ => {
                let left = 1;
                let right = 2;
                if right >= self.elements.len() || self.elements[left] <= self.elements[right] {
                    Some(&self.elements[left])
                } else {
                    Some(&self.elements[right])
                }
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn peek_max(&self) -> Option<&T> {
        self.elements.first()
    }

    pub fn pop_min_k(&mut self, k: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(k);
        for _ in 0..k {
            if let Some(min) = self.pop_min() {
                result.push(min);
            } else {
                break;
            }
        }
        result
    }

    pub fn push(&mut self, value: T) {
        self.elements.push(value);
        let index = self.elements.len() - 1;
        self.sift_up(index);
    }

    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = Self::to_parent(index);
            let current_level = Self::level(index);
            // Parent is bigger than current, Grandparent is smaller than parent
            if Self::is_max_level(current_level) {
                if self.elements[index] < self.elements[parent] {
                    self.elements.swap(index, parent);
                    index = parent;
                } else {
                    let grandparent = Self::to_parent(parent);
                    if grandparent != parent && self.elements[index] > self.elements[grandparent] {
                        self.elements.swap(index, grandparent);
                        index = grandparent;
                    } else {
                        break;
                    }
                }
            } else {
                // Parent is smaller than current, Grandparent is bigger than parent
                if self.elements[index] > self.elements[parent] {
                    self.elements.swap(index, parent);
                    index = parent;
                } else {
                    let grandparent = Self::to_parent(parent);
                    if grandparent != parent && self.elements[index] < self.elements[grandparent] {
                        self.elements.swap(index, grandparent);
                        index = grandparent;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    fn to_parent(index: usize) -> usize {
        if index == 0 {
            0
        } else {
            (index - 1) / 2
        }
    }

    fn level(index: usize) -> usize {
        (index + 1).ilog2() as usize
    }

    // the level is greater than the childrens
    fn is_max_level(level: usize) -> bool {
        level % 2 == 0
    }

    fn is_min_level(level: usize) -> bool {
        level % 2 == 1
    }

    pub fn pop_max(&mut self) -> Option<T> {
        if self.elements.is_empty() {
            return None;
        }
        let min = self.elements.swap_remove(0);
        if !self.elements.is_empty() {
            let current = self.sift_down_max(0);
            self.sift_up(current);
        }
        Some(min)
    }

    fn sift_down_min(&mut self, mut current: usize) -> usize {
        loop {
            let mut smallest = current;
            let left = 2 * current + 1;
            let right = 2 * current + 2;

            // Check children
            for &child in &[left, right] {
                if child < self.elements.len() && self.elements[child] < self.elements[smallest] {
                    smallest = child;
                }
            }

            // Check grandchildren
            let grandchildren = [2 * left + 1, 2 * left + 2, 2 * right + 1, 2 * right + 2];
            for &grandchild in &grandchildren {
                if grandchild < self.elements.len()
                    && self.elements[grandchild] < self.elements[smallest]
                {
                    smallest = grandchild;
                }
            }
            if smallest != current {
                self.elements.swap(current, smallest);
                current = smallest;
                if Self::is_min_level(Self::level(current)) {
                    continue;
                } else {
                    return self.sift_down_max(current);
                }
            } else {
                return smallest;
            }
        }
    }

    pub fn pop_min(&mut self) -> Option<T> {
        if self.elements.is_empty() {
            return None;
        }
        let min_index = match self.elements.len() {
            1 => 0,
            _ => {
                let left = 1;
                let right = 2;
                if right >= self.elements.len() || self.elements[left] <= self.elements[right] {
                    left
                } else {
                    right
                }
            }
        };
        let max = self.elements.swap_remove(min_index);
        if min_index < self.elements.len() {
            let current = self.sift_down_min(min_index);
            self.sift_up(current);
        }
        Some(max)
    }

    fn sift_down_max(&mut self, mut current: usize) -> usize {
        loop {
            let mut largest = current;
            let left = 2 * current + 1;
            let right = 2 * current + 2;

            // Check children
            for &child in &[left, right] {
                if child < self.elements.len() && self.elements[child] > self.elements[largest] {
                    largest = child;
                }
            }

            // Check grandchildren
            let grandchildren = [2 * left + 1, 2 * left + 2, 2 * right + 1, 2 * right + 2];
            for &grandchild in &grandchildren {
                if grandchild < self.elements.len()
                    && self.elements[grandchild] > self.elements[largest]
                {
                    largest = grandchild;
                }
            }
            if largest != current {
                self.elements.swap(current, largest);
                current = largest;
                if Self::is_max_level(Self::level(current)) {
                    continue;
                } else {
                    return self.sift_down_min(current);
                }
            } else {
                return largest;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use space::Neighbor;

    use super::*;

    #[test]
    fn test_basic() {
        let mut heap = MinMaxHeap::new();
        heap.push(3);
        heap.push(1);
        heap.push(2);
        assert_eq!(heap.peek_min(), Some(&1));
        assert_eq!(heap.peek_max(), Some(&3));
        assert_eq!(heap.pop_min(), Some(1));
        assert_eq!(heap.pop_max(), Some(3));
        assert_eq!(heap.pop_min(), Some(2));
        assert_eq!(heap.pop_min(), None);
    }

    #[test]
    fn test_complex() {
        let mut heap = MinMaxHeap::new();
        heap.push(4);
        heap.push(2);
        heap.push(6);
        heap.push(1);
        heap.push(5);
        assert_eq!(heap.peek_min(), Some(&1));
        assert_eq!(heap.peek_max(), Some(&6));
        assert_eq!(heap.pop_min(), Some(1));
        assert_eq!(heap.peek_min(), Some(&2));
        assert_eq!(heap.pop_max(), Some(6));
        assert_eq!(heap.peek_max(), Some(&5));

        let mut heap = MinMaxHeap::new();
        for i in 0..1000 {
            heap.push(i);
        }

        for i in 0..1000 {
            assert_eq!(heap.peek_min(), Some(&i));
            assert_eq!(heap.pop_min(), Some(i));
        }

        for i in 0..1000 {
            heap.push(i);
        }

        for i in 0..500 {
            assert_eq!(heap.pop_min(), Some(i));
            assert_eq!(heap.pop_max(), Some(1000 - i - 1));
        }

        for i in (0..1000).rev() {
            heap.push(i);
        }
        for i in 0..500 {
            assert_eq!(heap.pop_min(), Some(i));
            assert_eq!(heap.pop_max(), Some(1000 - i - 1));
        }

        let mut heap = MinMaxHeap::new();
        for i in 0..10 {
            heap.push(Neighbor {
                index: 10 - i,
                distance: i,
            });
        }
    }
}
