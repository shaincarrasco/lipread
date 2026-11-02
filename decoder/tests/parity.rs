//! Parity with the Python reference (src/lipread/lm.py, beam.py, decode.py), on the fixtures written by
//! tests_web/make_beam_fixtures.py (the same ones the JS port is tested against).
//!
//!   cd decoder && cargo test --release

use std::path::PathBuf;
use std::time::Instant;

use lipread_decoder::{Decoder, DecodeOptions};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}
fn fix(name: &str) -> PathBuf {
    root().join("tests_web").join("fixtures").join(name)
}
fn read_json(p: PathBuf) -> Value {
    serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}
fn decoder() -> Decoder {
    let lm = std::fs::read(root().join("web/core/assets/lm.bin")).unwrap();
    let vocab = std::fs::read_to_string(root().join("web/core/assets/tokens.json")).unwrap();
    Decoder::try_new(&lm, &vocab).unwrap()
}

/// Rebuild the dense (T, V) matrix exactly as make_beam_fixtures.py does: top-K stored, the rest `fill`.
fn dense(refs: &Value, clip: &Value) -> Vec<f32> {
    let t = clip["T"].as_u64().unwrap() as usize;
    let k = refs["K"].as_u64().unwrap() as usize;
    let v = refs["V"].as_u64().unwrap() as usize;
    let fill = refs["fill"].as_f64().unwrap() as f32;
    let raw = std::fs::read(fix(clip["file"].as_str().unwrap())).unwrap();
    let ib = t * k * 2;
    assert_eq!(raw.len(), ib + t * k * 4);
    let mut d = vec![fill; t * v];
    for f in 0..t {
        for j in 0..k {
            let n = f * k + j;
            let idx = u16::from_le_bytes([raw[2 * n], raw[2 * n + 1]]) as usize;
            let o = ib + 4 * n;
            d[f * v + idx] = f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
        }
    }
    d
}

#[test]
fn lm_header_and_rejects_garbage() {
    let d = decoder();
    let lm = d.lm().unwrap();
    assert_eq!(lm.header.order, Some(3));
    assert_eq!(lm.words.len(), lm.header.v);
    assert_eq!(lm.words[0], "<s>");
    assert_eq!(lm.words[2], "<unk>");
    assert!(Decoder::try_new(&[0u8; 64], "[\"a\"]").err().unwrap().contains("not a LipRead LM file"));
}

#[test]
fn lm_scores_equal_python_triples() {
    let d = decoder();
    let lm = d.lm().unwrap();
    let f = read_json(fix("beam_lm_triples.json"));
    assert_eq!(f["V"].as_u64().unwrap() as usize, lm.header.v);
    let triples = f["triples"].as_array().unwrap();
    let scores = f["scores"].as_array().unwrap();
    assert_eq!(triples.len(), 2000);
    let opt = |x: &Value| x.as_u64().map(|n| n as u32);
    let mut worst = 0.0f64;
    for (i, tr) in triples.iter().enumerate() {
        let w = tr[0].as_u64().unwrap() as u32;
        let s = lm.score(w, opt(&tr[1]), opt(&tr[2]));
        let r = scores[i].as_f64().unwrap();
        worst = worst.max((s - r).abs());
        assert!((s - r).abs() < 1e-9, "triple {i} {tr}: {s} vs {r}");
    }
    eprintln!("LM: max |Rust - Python| over {} triples: {worst:e}", triples.len());
    for pair in f["word_ids"].as_array().unwrap() {
        assert_eq!(lm.id(pair[0].as_str().unwrap()) as u64, pair[1].as_u64().unwrap(), "id({})", pair[0]);
    }
}

fn texts(ws: &Value) -> Vec<String> {
    ws.as_array().unwrap().iter().map(|w| w["text"].as_str().unwrap().to_string()).collect()
}

#[test]
fn beam_equals_python_on_all_fixtures() {
    let mut d = decoder();
    let refs = read_json(fix("beam_refs.json"));
    let v = refs["V"].as_u64().unwrap() as usize;
    let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
    let (mut n, mut frames, mut total_ms, mut worst) = (0, 0usize, 0.0f64, 0.0f64);
    for clip in refs["clips"].as_array().unwrap() {
        let lp = dense(&refs, clip);
        let t = clip["T"].as_u64().unwrap() as usize;
        let t0 = clip["t0"].as_f64().unwrap();
        for (set, opts) in refs["option_sets"].as_object().unwrap() {
            let id = format!("{} / {set}", clip["id"].as_str().unwrap());
            let o: DecodeOptions = serde_json::from_value(opts.clone()).unwrap();
            let json = d.decode_json(&lp, t, v, t0, &opts.to_string()).unwrap();
            let out: Value = serde_json::from_str(&json).unwrap();
            let ow = out["words"].as_array().unwrap();
            let rw = clip["results"][set]["words"].as_array().unwrap();
            assert_eq!(texts(&out["words"]), texts(&clip["results"][set]["words"]), "{id}: word texts");
            for (i, (w, r)) in ow.iter().zip(rw).enumerate() {
                assert_eq!(texts(&w["alternatives"]), texts(&r["alternatives"]), "{id}: word {i} alternative texts");
                assert_eq!(w["tier"], r["tier"], "{id}: word {i} tier");
                for key in ["start", "end", "confidence", "p"] {
                    let (a, b) = (w[key].as_f64().unwrap(), r[key].as_f64().unwrap());
                    assert!(near(a, b), "{id}: word {i} {key} {a} vs {b}");
                    worst = worst.max((a - b).abs());
                }
                for (j, (a, b)) in w["alternatives"].as_array().unwrap().iter().zip(r["alternatives"].as_array().unwrap()).enumerate() {
                    let (a, b) = (a["p"].as_f64().unwrap(), b["p"].as_f64().unwrap());
                    assert!(near(a, b), "{id}: word {i} alt {j} p {a} vs {b}");
                }
                assert_eq!(w["boosted"], Value::Bool(false));
                assert!(w["original"].is_null());
            }
            // per-word alternative lists, as the Python decoder returns them separately
            let ra = clip["results"][set]["alternatives"].as_array().unwrap();
            assert_eq!(ra.len(), ow.len(), "{id}");
            for (w, a) in ow.iter().zip(ra) {
                assert_eq!(texts(&w["alternatives"]), texts(&a["alternatives"]), "{id}");
                assert!(near(w["p"].as_f64().unwrap(), a["p"].as_f64().unwrap()), "{id}");
            }
            // timing (best of 3, warm)
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let s = Instant::now();
                d.decode_words(&lp, t, v, t0, &o).unwrap();
                best = best.min(s.elapsed().as_secs_f64() * 1e3);
            }
            total_ms += best;
            frames += t;
            n += 1;
        }
    }
    assert_eq!(n, 40);
    eprintln!("beam: {n} decodes equal Python (max |diff| of start/end/confidence/p: {worst:e}); native {:.4} ms/frame ({total_ms:.1} ms total)", total_ms / frames as f64);
}

#[test]
fn greedy_equals_python() {
    let d = decoder();
    for name in ["booster_case0", "booster_case3"] {
        let meta = read_json(fix(&format!("{name}.json")));
        let raw = std::fs::read(fix(meta["logprobs_file"].as_str().unwrap())).unwrap();
        let lp: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let t = meta["T"].as_u64().unwrap() as usize;
        let v = meta["V"].as_u64().unwrap() as usize;
        let out: Value = serde_json::from_str(&d.greedy_json(&lp, t, v, meta["t0"].as_f64().unwrap(), 0).unwrap()).unwrap();
        let ow = out["words"].as_array().unwrap();
        let pw = meta["greedy"].as_array().unwrap();
        assert_eq!(ow.len(), pw.len(), "{name}");
        for (w, p) in ow.iter().zip(pw) {
            assert_eq!(w["text"], p["text"]);
            assert_eq!(w["tier"], p["tier"]);
            for (key, tol) in [("start", 1e-9), ("end", 1e-9), ("confidence", 1e-5)] {
                let (a, b) = (w[key].as_f64().unwrap(), p[key].as_f64().unwrap());
                assert!((a - b).abs() < tol, "{name} {key}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn options_defaults_match_python() {
    let o = Decoder::parse_options("").unwrap();
    assert_eq!((o.beam, o.topk, o.n_alternatives, o.blank), (16, 24, 3, 0));
    assert_eq!((o.alpha, o.beta, o.unk_penalty, o.prune, o.posterior_scale, o.temperature), (0.3, 1.0, -4.0, 10.0, 0.2, 1.0));
    assert_eq!(o.name_logp, (1e-4f64).ln());
    assert!(o.names.is_empty());
    let o = Decoder::parse_options(r#"{"beam": 64, "n_alternatives": 4, "unknown": 1}"#).unwrap();
    assert_eq!((o.beam, o.n_alternatives, o.topk), (64, 4, 24));
}

#[test]
fn round4_is_python_round() {
    use lipread_decoder::beam::round4;
    assert_eq!(round4(0.03125), 0.0312); // exact tie -> even
    assert_eq!(round4(0.09375), 0.0938);
    assert_eq!(round4(0.12345), 0.1235); // 0.12345 is slightly above the tie in binary
    assert_eq!(round4(0.955), 0.955);
    assert_eq!(round4(1.0), 1.0);
    assert_eq!(round4(0.0), 0.0);
}

#[test]
fn decodes_without_lm_and_rejects_bad_shapes() {
    let vocab = std::fs::read_to_string(root().join("web/core/assets/tokens.json")).unwrap();
    let mut d = Decoder::try_new(&[], &vocab).unwrap();
    assert!(d.lm().is_none());
    let v = d.vocab().len();
    assert!(d.decode_json(&vec![0.0; 10], 1, v, 0.0, "").is_err());
    assert!(d.decode_json(&vec![0.0; v], 1, v + 1, 0.0, "").is_err());
    let mut lp = vec![-50.0f32; 3 * v];
    lp[0] = 0.0; // blank everywhere
    lp[v] = 0.0;
    lp[2 * v] = 0.0;
    assert_eq!(d.decode_json(&lp, 3, v, 0.0, "").unwrap(), r#"{"words":[]}"#);
}
