# Research Log

Chronological record of ingests, queries, and lint passes.

Format: `## [YYYY-MM-DD] <type> | <subject>` where `<type>` is one of: `ingest | query | lint | scaffold`.

To get the last N entries: `grep "^## \[" log.md | tail -N`.

---

## [2026-05-23] scaffold | Initial wiki structure

- Created `research/` following the Karpathy LLM-wiki pattern.
- Schema: `CLAUDE.md` with ingest/query/lint operations.
- Layout: `raw/`, `wiki/{sources,concepts,entities,questions}`, `index.md`, `log.md`, `overview.md`.
- Bootstrapped initial research questions in `wiki/questions/`:
  - `bioes-vs-span-based.md`
  - `person-detection-failure-modes.md`
  - `synthetic-data-quality.md`
  - `initial-reading-list.md` — ~15 papers to ingest first via paper7.
- No sources ingested yet.
