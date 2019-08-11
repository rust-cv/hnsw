#[derive(Clone, Debug, Default)]
pub struct Candidates {
    candidates: Vec<(f32, u32)>,
}

impl Candidates {
    /// Clears the struct without freeing the memory.
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// Pushes a new node to the candidate list.
    pub fn push(&mut self, distance: f32, node: u32) {
        let unsigned_distance: u32 = unsafe { std::mem::transmute(distance) };
        let pos = self
            .candidates
            .binary_search_by(|&(d, _)| {
                let unsigned_d: u32 = unsafe { std::mem::transmute(d) };
                unsigned_distance.cmp(&unsigned_d)
            })
            .unwrap_or_else(|e| e);
        self.candidates.insert(pos, (distance, node));
    }

    /// Pop an item from the candidate list.
    pub fn pop(&mut self) -> Option<(f32, u32)> {
        self.candidates.pop()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FixedCandidates {
    candidates: Vec<(f32, u32)>,
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

    /// Pushes a new node to the candidate list.
    pub fn push(&mut self, distance: f32, node: u32) -> bool {
        let unsigned_distance: u32 = unsafe { std::mem::transmute(distance) };
        let better = self
            .candidates
            .last()
            .map(|&(df, _)| {
                let last_distance: u32 = unsafe { std::mem::transmute(df) };
                unsigned_distance < last_distance
            })
            .unwrap_or(self.cap != 0);
        let full = self.len() == self.cap;
        let will_add = better | !full;
        if will_add {
            let pos = self
                .candidates
                .binary_search_by(|&(d, _)| {
                    let unsigned_d: u32 = unsafe { std::mem::transmute(d) };
                    unsigned_d.cmp(&unsigned_distance)
                })
                .unwrap_or_else(|e| e);
            self.candidates.insert(pos, (distance, node));
            if full {
                self.pop();
            }
        }

        will_add
    }

    /// Pop the worst item from the candidate list.
    pub fn pop(&mut self) -> Option<(f32, u32)> {
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
    assert!(candidates.push(1.0, 0));
    assert!(candidates.push(0.5, 1));
    assert!(candidates.push(0.000_000_01, 2));
    assert!(!candidates.push(1.1, 3));
    assert!(!candidates.push(2.0, 4));
    assert!(candidates.push(0.000_000_000_1, 5));
    assert!(!candidates.push(1_000_000.0, 6));
    assert!(!candidates.push(0.6, 7));
    assert!(!candidates.push(0.5, 8));
    assert_eq!(
        &candidates.candidates,
        &[(0.000_000_000_1, 5), (0.000_000_01, 2), (0.5, 1)]
    );
    assert!(candidates.push(0.000_000_01, 9));
    assert!(!candidates.push(0.000_000_01, 10));
    let mut arr = [0; 3];
    candidates.fill_slice(&mut arr);
    assert!(arr == [5, 9, 2] || arr == [9, 5, 2]);
}
