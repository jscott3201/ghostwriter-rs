//! `gw-storage` — the reproducible data plane.
//!
//! SQLite (`rusqlite`, bundled) for run/queue/provenance/lifecycle state, and Arrow/Parquet
//! for the byte-reproducible admitted-record export. The `Store` trait is kept abstract so a
//! provenance-graph backend can slot in later. (The vector index — `usearch` — and embedding
//! decontamination are added with the storage implementation.)
