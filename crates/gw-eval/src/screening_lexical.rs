//! Exact token tuples, bounded reusable shingle indexes, and canonical pair evidence.
use gw_schema::{LexicalScreeningEvidence, ScreeningField, ScreeningLimits, ScreeningPolicy};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(crate) struct Segment {
    pub id: String,
    pub field: ScreeningField,
    pub tokens: Vec<u32>,
}

#[derive(Default)]
pub(crate) struct TextIndex {
    pub segments: Vec<Segment>,
    words: BTreeMap<String, u32>,
    sets: BTreeMap<(usize, u32), Vec<usize>>,
    text_bytes: u64,
    shingles: u64,
    shingle_token_work: u64,
    comparisons: u64,
}

fn separator(character: char) -> bool {
    matches!(character, '\u{0009}'..='\u{000d}' | '\u{0020}' | '\u{0085}' | '\u{00a0}'
        | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}'
        | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn add_count(counter: &mut u64, count: u64, limit: u64) -> Result<(), &'static str> {
    let next = counter
        .checked_add(count)
        .ok_or("resource_counter_overflow")?;
    if next > limit {
        return Err("resource_limit");
    }
    *counter = next;
    Ok(())
}

impl TextIndex {
    pub fn add(
        &mut self,
        id: String,
        field: ScreeningField,
        text: &str,
        limits: &ScreeningLimits,
    ) -> Result<usize, &'static str> {
        if text.len() as u64 > limits.segment_bytes {
            return Err("segment_bytes_limit");
        }
        if self.segments.len() as u64 >= limits.segments {
            return Err("segment_count_limit");
        }
        add_count(
            &mut self.text_bytes,
            text.len() as u64,
            limits.total_text_bytes,
        )
        .map_err(|_| "total_text_bytes_limit")?;
        let mut tokens = Vec::new();
        for word in text.split(separator).filter(|word| !word.is_empty()) {
            if tokens.len() as u64 >= limits.segment_tokens {
                return Err("segment_tokens_limit");
            }
            let word = word.to_ascii_lowercase();
            let next = u32::try_from(self.words.len()).map_err(|_| "token_identity_overflow")?;
            tokens.push(*self.words.entry(word).or_insert(next));
        }
        let position = self.segments.len();
        self.segments.push(Segment { id, field, tokens });
        Ok(position)
    }

    pub fn comparison(&mut self, policy: &ScreeningPolicy) -> Result<(), &'static str> {
        add_count(&mut self.comparisons, 1, policy.limits.comparisons)
            .map_err(|_| "comparison_limit")
    }

    pub fn build(&mut self, policy: &ScreeningPolicy) -> Result<(), &'static str> {
        for segment in 0..self.segments.len() {
            let length = self.segments[segment].tokens.len() as u32;
            let mut lengths = BTreeSet::new();
            for n in policy.ngram[0]..=policy.ngram[1].min(length) {
                lengths.insert(n);
            }
            if policy.min_overlap_tokens <= length {
                lengths.insert(policy.min_overlap_tokens);
            }
            for n in lengths {
                // Keys borrow token slices during construction. Persist only starting positions;
                // even a very long shingle never allocates a second copy of its token tuple.
                let tokens = &self.segments[segment].tokens;
                let windows = (tokens.len() - n as usize + 1) as u64;
                let work = windows
                    .checked_mul(u64::from(n))
                    .ok_or("resource_counter_overflow")?;
                add_count(
                    &mut self.shingle_token_work,
                    work,
                    policy.limits.shingle_token_work,
                )
                .map_err(|_| "shingle_token_work_limit")?;
                let mut distinct = BTreeMap::new();
                for (start, tuple) in tokens.windows(n as usize).enumerate() {
                    if !distinct.contains_key(tuple) {
                        add_count(&mut self.shingles, 1, policy.limits.distinct_shingles)
                            .map_err(|_| "distinct_shingles_limit")?;
                        distinct.insert(tuple, start);
                    }
                }
                self.sets
                    .insert((segment, n), distinct.into_values().collect());
            }
        }
        Ok(())
    }

    fn overlap(&self, left: usize, right: usize, n: u32) -> (u64, u64) {
        let (Some(a), Some(b)) = (self.sets.get(&(left, n)), self.sets.get(&(right, n))) else {
            return (0, 0);
        };
        let (mut i, mut j, mut intersection) = (0, 0, 0);
        let (a_tokens, b_tokens) = (&self.segments[left].tokens, &self.segments[right].tokens);
        while i < a.len() && j < b.len() {
            match a_tokens[a[i]..a[i] + n as usize].cmp(&b_tokens[b[j]..b[j] + n as usize]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        (intersection, a.len() as u64 + b.len() as u64 - intersection)
    }

    pub fn matches(
        &mut self,
        left: usize,
        right: usize,
        policy: &ScreeningPolicy,
    ) -> Result<Vec<LexicalScreeningEvidence>, &'static str> {
        self.comparison(policy)?;
        if self.overlap(left, right, policy.min_overlap_tokens).0 == 0 {
            return Ok(vec![]);
        }
        let mut evidence = Vec::new();
        for n in policy.ngram[0]
            ..=policy.ngram[1]
                .min(self.segments[left].tokens.len() as u32)
                .min(self.segments[right].tokens.len() as u32)
        {
            let (intersection, union) = self.overlap(left, right, n);
            // Stored distinct counts are bounded below 2^53, making integer conversion exact.
            if intersection > 0 && (intersection as f64 / union as f64) >= policy.jaccard_threshold
            {
                evidence.push(LexicalScreeningEvidence {
                    left: self.segments[left].id.clone(),
                    right: self.segments[right].id.clone(),
                    n: Some(n),
                    intersection,
                    union,
                });
            }
        }
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_whitespace_and_ascii_case_only() {
        let mut index = TextIndex::default();
        let limits = ScreeningLimits::default();
        let a=index.add("a".into(),ScreeningField::Content," A\tB\nC\u{85}D\u{a0}E\u{1680}F\u{2000}G\u{200a}H\u{2028}I\u{2029}J\u{202f}K\u{205f}L\u{3000}M ",&limits).unwrap();
        let b = index
            .add(
                "b".into(),
                ScreeningField::Content,
                "a b c d e f g h i j k l m",
                &limits,
            )
            .unwrap();
        assert_eq!(index.segments[a].tokens, index.segments[b].tokens);
        for (x, y) in [
            ("1", "+1"),
            ("x!", "x"),
            ("É", "é"),
            ("é", "e\u{301}"),
            ("a\u{200b}b", "a b"),
            ("a\u{1c}b", "a b"),
        ] {
            let x = index
                .add("x".into(), ScreeningField::Content, x, &limits)
                .unwrap();
            let y = index
                .add("y".into(), ScreeningField::Content, y, &limits)
                .unwrap();
            assert_ne!(index.segments[x].tokens, index.segments[y].tokens);
        }
    }
    #[test]
    fn independent_set_cardinality_any_n_and_inclusive_threshold() {
        let mut index = TextIndex::default();
        let mut policy = ScreeningPolicy {
            ngram: [2, 3],
            min_overlap_tokens: 2,
            jaccard_threshold: 0.5,
            ..Default::default()
        };
        let a = index
            .add(
                "a".into(),
                ScreeningField::Content,
                "a b a b",
                &policy.limits,
            )
            .unwrap();
        let b = index
            .add("b".into(), ScreeningField::Content, "a b c", &policy.limits)
            .unwrap();
        index.build(&policy).unwrap();
        // 2-grams are {ab,ba} and {ab,bc}: |intersection|=1, |union|=3.
        assert!(index.matches(a, b, &policy).unwrap().is_empty());
        policy.jaccard_threshold = 1.0 / 3.0;
        let found = index.matches(a, b, &policy).unwrap();
        assert_eq!(
            found
                .iter()
                .map(|m| (m.n, m.intersection, m.union))
                .collect::<Vec<_>>(),
            vec![(Some(2), 1, 3)]
        );
        policy.jaccard_threshold = f64::from_bits((1.0_f64 / 3.0).to_bits() + 1);
        assert!(index.matches(a, b, &policy).unwrap().is_empty());
    }
    #[test]
    fn repeated_windows_consume_work_even_when_distinct_shingles_are_few() {
        let mut index = TextIndex::default();
        let mut policy = ScreeningPolicy {
            ngram: [1, 4],
            min_overlap_tokens: 1,
            ..Default::default()
        };
        policy.limits.shingle_token_work = 29;
        index
            .add(
                "repeated".into(),
                ScreeningField::Content,
                "a a a a a a",
                &policy.limits,
            )
            .unwrap();
        // Windows cost 6*1 + 5*2 + 4*3 + 3*4 = 40 token positions, despite only four
        // distinct tuples across all n. At 29, the fourth length must fail before iteration.
        assert_eq!(index.build(&policy), Err("shingle_token_work_limit"));
    }
    #[test]
    fn resource_counter_overflow_is_explicit() {
        let mut counter = u64::MAX;
        assert_eq!(
            add_count(&mut counter, 1, u64::MAX),
            Err("resource_counter_overflow")
        );
        assert_eq!(counter, u64::MAX);
    }
}
