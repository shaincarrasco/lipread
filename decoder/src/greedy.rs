//! Greedy CTC decoding into timed, confidence-scored words (port of `decode.greedy_words`).
//!
//! The frame-wise argmax (first index on ties, as numpy) is collapsed into runs of one emitting piece;
//! a silent frame (blank or special token) ends a run. Runs are grouped into words at pieces that start
//! with "▁". Confidence is the mean posterior of the winning piece over the word's frames (in f64).

use serde::Serialize;

use crate::beam::{tier_for, FPS, MARK};

/// A greedy word, with the same fields as the JS `greedyWords` output.
#[derive(Serialize, Debug, Clone)]
pub struct GreedyWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub tier: &'static str,
    pub boosted: bool,
    pub original: Option<String>,
}

pub fn greedy_words(lp: &[f32], t_len: usize, v: usize, vocab: &[String], t0: f64, blank: usize) -> Result<Vec<GreedyWord>, String> {
    if v != vocab.len() {
        return Err(format!("logprobs must have {} columns, got {v}", vocab.len()));
    }
    if lp.len() < t_len * v {
        return Err(format!("logprobs too short for T={t_len}, V={v}"));
    }
    let silent: Vec<bool> =
        vocab.iter().enumerate().map(|(i, p)| i == blank || (p.starts_with('<') && p.ends_with('>'))).collect();

    // (piece, first_frame, last_frame), inclusive
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut prev: Option<usize> = None;
    for t in 0..t_len {
        let row = &lp[t * v..(t + 1) * v];
        let (mut k, mut best) = (0usize, f32::NEG_INFINITY);
        for (j, &x) in row.iter().enumerate() {
            if x > best {
                best = x;
                k = j;
            }
        }
        if silent[k] {
            prev = None;
            continue;
        }
        if prev == Some(k) {
            runs.last_mut().unwrap().2 = t;
        } else {
            runs.push((k, t, t));
            prev = Some(k);
        }
    }

    let mut groups: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for r in runs {
        if vocab[r.0].starts_with(MARK) || groups.is_empty() {
            groups.push(vec![r]);
        } else {
            groups.last_mut().unwrap().push(r);
        }
    }

    let mut words = Vec::new();
    for g in &groups {
        let joined: String = g.iter().map(|&(k, _, _)| vocab[k].replace(MARK, "")).collect();
        let text = joined.trim();
        if text.is_empty() {
            continue;
        }
        let (mut sum, mut n) = (0.0f64, 0usize);
        for &(k, a, b) in g {
            for t in a..=b {
                sum += (lp[t * v + k] as f64).exp();
                n += 1;
            }
        }
        let confidence = sum / n as f64;
        words.push(GreedyWord {
            text: text.to_string(),
            start: t0 + g[0].1 as f64 / FPS,
            end: t0 + (g[g.len() - 1].2 + 1) as f64 / FPS,
            confidence,
            tier: tier_for(confidence),
            boosted: false,
            original: None,
        });
    }
    Ok(words)
}
