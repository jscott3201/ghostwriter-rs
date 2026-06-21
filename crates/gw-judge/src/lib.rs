//! `gw-judge` — two-rail grading and admission.
//!
//! Rail 1: a deterministic verifier (RLVR) as the authoritative hard gate where ground truth
//! exists. Rail 2: an LLM judge panel elsewhere. **Load-bearing piece:** the weighted
//! design-effect consensus `effective_n = (Σw)² / (wᵀ R w)` over sealed verdicts — never
//! naive majority/mean. Depends on `gw-schema`, `gw-providers`, and `gw-storage`.
