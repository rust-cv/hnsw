#[cfg(feature = "serde-impl")]
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde-impl", derive(Serialize, Deserialize))]
pub struct Candidates {
    candidates: Vec<(u32, u32)>,
}

impl Candidates {
    /// Clears the struct without freeing the memory.
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// Pushes a new node to the candidate list.
    pub fn push(&mut self, distance: u32, node: u32) {
        let pos = self
            .candidates
            .binary_search_by_key(&distance, |&(d, _)| d)
            .unwrap_or_else(|e| e);
        self.candidates.insert(pos, (distance, node));
    }

    /// Pop an item from the candidate list.
    pub fn pop(&mut self) -> Option<(u32, u32)> {
        self.candidates.pop()
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde-impl", derive(Serialize, Deserialize))]
pub struct FixedCandidates {
    candidates: Vec<(u32, u32)>,
    cap: usize,
}

impl FixedCandidates {
    /// Clears the struct without freeing the memory.
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.cap = 0;
    }

    /// Gets the number of items in the candidate pool.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Checks if any candidates are present.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pushes a new node to the candidate list.
    pub fn push(&mut self, distance: u32, node: u32) -> bool {
        let better = self
            .candidates
            .last()
            .map(|&(df, _)| distance < df)
            .unwrap_or(self.cap != 0);
        let full = self.len() == self.cap;
        let will_add = better | !full;
        if will_add {
            let pos = self
                .candidates
                .binary_search_by(|&(d, _)| d.cmp(&distance))
                .unwrap_or_else(|e| e);
            self.candidates.insert(pos, (distance, node));
            if full {
                self.pop();
            }
        }

        will_add
    }

    /// Pop the worst item from the candidate list.
    pub fn pop(&mut self) -> Option<(u32, u32)> {
        self.candidates.pop()
    }

    /// Sets the cap to `cap`. Resizes if necessary, removing the bottom elements.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        if self.candidates.len() > cap {
            // Remove the bottom items.
            self.candidates.drain(0..self.candidates.len() - cap);
        }
    }

    /// Fill a slice with the best elements and return the part of the slice written.
    pub fn fill_slice<'a>(&self, s: &'a mut [u32]) -> &'a mut [u32] {
        let total_fill = std::cmp::min(s.len(), self.len());
        for (ix, node) in self
            .candidates
            .iter()
            .map(|(_, n)| n)
            .cloned()
            .take(total_fill)
            .enumerate()
        {
            s[ix] = node;
        }
        &mut s[0..total_fill]
    }
}

#[cfg(test)]
#[test]
fn test_candidates() {
    let mut candidates = FixedCandidates::default();
    candidates.set_cap(3);
    assert!(candidates.push(1.0f32.to_bits(), 0));
    assert!(candidates.push(0.5f32.to_bits(), 1));
    assert!(candidates.push(0.000_000_01f32.to_bits(), 2));
    assert!(!candidates.push(1.1f32.to_bits(), 3));
    assert!(!candidates.push(2.0f32.to_bits(), 4));
    assert!(candidates.push(0.000_000_000_1f32.to_bits(), 5));
    assert!(!candidates.push(1_000_000.0f32.to_bits(), 6));
    assert!(!candidates.push(0.6f32.to_bits(), 7));
    assert!(!candidates.push(0.5f32.to_bits(), 8));
    assert!(candidates.push(0.000_000_01f32.to_bits(), 9));
    assert!(!candidates.push(0.000_000_01f32.to_bits(), 10));
    let mut arr = [0; 3];
    candidates.fill_slice(&mut arr);
    arr[0..2].sort_unstable();
    assert_eq!(arr, [5, 9, 2]);
}
