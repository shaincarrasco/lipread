/-!
# Machine-checked proof: long videos are split into windows with no gaps

The lip model takes at most 500 frames at a time (20 s of video). `lipread.engine.stitch_windows`
(and its JS twin `stitchWindows` in `web/core/lipread.js`) cuts a longer video into overlapping
windows and stitches the model's per-frame outputs back together:

```python
step   = chunk - overlap
starts = list(range(0, max(T - overlap, 1), step))
if starts[-1] + chunk < T:
    starts.append(T - chunk)
# window s covers frames [s, min(s + chunk, T))
```

If some frame were in no window, that moment of the video would silently get no transcript.
This file proves, for every video length `T > chunk` and every `overlap < chunk`:

* `every_frame_covered`  every frame `t < T` lies inside some window;
* `windows_in_range`     every window starts inside the video (no empty or out-of-range window);
* `range_alone_suffices` the `range(...)` windows already cover everything, so the
                         `starts.append(T - chunk)` line is a safety net that never fires.

Check it with `lean proofs/Windows.lean` (Lean 4 core only, no Mathlib).
-/

namespace LipRead

/-- Python's `range(0, n, step)` (for `step ≥ 1`): `ceil(n / step)` multiples of `step`. -/
def pyRange (n step : Nat) : List Nat :=
  (List.range ((n + step - 1) / step)).map (· * step)

/-- The `range(...)` part of the window starts. -/
def baseStarts (T chunk overlap : Nat) : List Nat :=
  pyRange (max (T - overlap) 1) (chunk - overlap)

/-- All window starts, exactly as `stitch_windows` computes them. -/
def starts (T chunk overlap : Nat) : List Nat :=
  let b := baseStarts T chunk overlap
  match b.getLast? with
  | some l => if l + chunk < T then b ++ [T - chunk] else b
  | none   => b

/-- Window `s` covers frame `t`. -/
def Covers (T chunk s t : Nat) : Prop := s ≤ t ∧ t < min (s + chunk) T

theorem mem_pyRange {n step s : Nat} :
    s ∈ pyRange n step ↔ ∃ k, k < (n + step - 1) / step ∧ k * step = s := by
  simp [pyRange, List.mem_map, List.mem_range]

/-- Ceiling division facts: `m * step ≥ n` and `m * step ≤ n + step - 1`. -/
theorem ceil_div_bounds (n step : Nat) (hs : 0 < step) :
    n ≤ (n + step - 1) / step * step ∧ (n + step - 1) / step * step ≤ n + step - 1 := by
  have h1 := Nat.div_add_mod (n + step - 1) step
  have h2 := Nat.mod_lt (n + step - 1) hs
  rw [Nat.mul_comm] at h1
  constructor <;> omega

theorem base_subset_starts (T chunk overlap s : Nat) (h : s ∈ baseStarts T chunk overlap) :
    s ∈ starts T chunk overlap := by
  unfold starts
  split
  · split
    · exact List.mem_append_left _ h
    · exact h
  · exact h

/-- The `range(...)` windows alone cover every frame. -/
theorem range_alone_suffices (T chunk overlap t : Nat)
    (hT : chunk < T) (hov : overlap < chunk) (ht : t < T) :
    ∃ s ∈ baseStarts T chunk overlap, Covers T chunk s t := by
  have hstep : 0 < chunk - overlap := by omega
  have hn : max (T - overlap) 1 = T - overlap := by omega
  -- name the quantities so `omega` only sees linear arithmetic
  generalize hstepv : chunk - overlap = step at hstep
  generalize hm : (T - overlap + step - 1) / step = m
  have ⟨hlo, hhi⟩ := ceil_div_bounds (T - overlap) step hstep
  rw [hm] at hlo hhi
  have hq := Nat.div_add_mod t step
  have hr := Nat.mod_lt t step hstep
  rw [Nat.mul_comm] at hq
  have hm1 : 1 ≤ m := by
    rcases Nat.eq_zero_or_pos m with h0 | h0
    · rw [h0] at hlo; omega
    · exact h0
  have hlast : (m - 1) * step = m * step - step := Nat.sub_one_mul m step
  by_cases hk : t / step ≤ m - 1
  · -- the window that starts at the last multiple of `step` at or before `t`
    have hle : t / step * step ≤ (m - 1) * step := Nat.mul_le_mul_right step hk
    refine ⟨t / step * step, ?_, ?_⟩
    · unfold baseStarts; rw [hn, hstepv, mem_pyRange, hm]
      exact ⟨t / step, by omega, rfl⟩
    · unfold Covers; constructor <;> omega
  · -- `t` is past the last `range` start: that window still reaches the end of the video
    have hge : m * step ≤ t / step * step :=
      Nat.mul_le_mul_right step (by omega)
    refine ⟨(m - 1) * step, ?_, ?_⟩
    · unfold baseStarts; rw [hn, hstepv, mem_pyRange, hm]
      exact ⟨m - 1, by omega, rfl⟩
    · unfold Covers; constructor <;> omega

/-- Every frame of the video is inside at least one window. -/
theorem every_frame_covered (T chunk overlap t : Nat)
    (hT : chunk < T) (hov : overlap < chunk) (ht : t < T) :
    ∃ s ∈ starts T chunk overlap, Covers T chunk s t := by
  obtain ⟨s, hs, hc⟩ := range_alone_suffices T chunk overlap t hT hov ht
  exact ⟨s, base_subset_starts T chunk overlap s hs, hc⟩

/-- Every window starts inside the video, so none is empty or out of range. -/
theorem windows_in_range (T chunk overlap s : Nat)
    (hT : chunk < T) (hov : overlap < chunk) (hs : s ∈ starts T chunk overlap) : s < T := by
  have hstep : 0 < chunk - overlap := by omega
  have hn : max (T - overlap) 1 = T - overlap := by omega
  have base : ∀ x ∈ baseStarts T chunk overlap, x < T := by
    intro x hx
    unfold baseStarts at hx
    rw [hn, mem_pyRange] at hx
    obtain ⟨k, hk, rfl⟩ := hx
    generalize hstepv : chunk - overlap = step at hstep hk ⊢
    generalize hm : (T - overlap + step - 1) / step = m at hk
    have ⟨_, hhi⟩ := ceil_div_bounds (T - overlap) step hstep
    rw [hm] at hhi
    have hle : k * step ≤ (m - 1) * step := Nat.mul_le_mul_right step (by omega)
    have hlast : (m - 1) * step = m * step - step := Nat.sub_one_mul m step
    omega
  unfold starts at hs
  split at hs
  · split at hs
    · rcases List.mem_append.mp hs with h | h
      · exact base s h
      · simp at h; omega
    · exact base s hs
  · exact base s hs

/-- The production constants: 500-frame windows with 50 frames of overlap. -/
example (T t : Nat) (hT : 500 < T) (ht : t < T) :
    ∃ s ∈ starts T 500 50, Covers T 500 s t :=
  every_frame_covered T 500 50 t hT (by decide) ht

end LipRead
