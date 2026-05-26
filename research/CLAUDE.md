# Research Wiki — Schema & Conventions

This directory is an LLM-maintained research wiki following the Karpathy "LLM wiki" pattern (https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

**Goal:** build a persistent, compounding knowledge base about NER for PII detection in Brazilian Portuguese, to inform improvements to the `privacy-filter-br` model and the `br-pii-guardrail` library.

## Directory layout

```
research/
├── CLAUDE.md            ← this file (schema)
├── index.md             ← catalog of every wiki page (LLM updates on each ingest)
├── log.md               ← chronological record (append-only, ## [YYYY-MM-DD] entries)
├── overview.md          ← running thesis + open questions
├── raw/                 ← immutable source documents (papers as markdown via paper7)
└── wiki/
    ├── sources/         ← one .md per ingested paper/article (summary, takeaways)
    ├── concepts/        ← named ideas (BIOES, distant supervision, span-based NER, ...)
    ├── entities/        ← models, datasets, benchmarks, orgs, people
    └── questions/       ← open research questions tied to our model's gaps
```

## Page conventions

Every wiki page starts with YAML frontmatter:

```yaml
---
type: source | concept | entity | question
tags: [comma-separated]
sources: <count of /raw/ files this page synthesizes from>
updated: YYYY-MM-DD
---
```

Use `[[page-name]]` to link between pages (Obsidian-style). A `[[name]]` that doesn't exist yet is fine — it's a placeholder for future work.

## Operations

### Ingest (add new paper)

When a new source arrives (e.g. via `paper7 fetch <arxiv-id>`):

1. **Read** the source from `raw/<id>.md` (paper7 output).
2. **Discuss** key takeaways with the user before writing.
3. **Create/update** the source page in `wiki/sources/<YYYY-MM-DD>-<slug>.md`:
   - Title, authors, year, venue
   - Abstract (1 paragraph, paraphrased)
   - Key findings (bullets)
   - How it relates to our work
   - Open questions raised
4. **Update relevant pages** in `wiki/concepts/` and `wiki/entities/` — cross-reference, add evidence, flag contradictions with prior sources.
5. **Update `index.md`** with the new page link.
6. **Append to `log.md`**:
   ```
   ## [2026-05-23] ingest | Title of Paper
   - paper7 id: 2407.12345
   - touched: wiki/sources/2026-05-23-foo.md, wiki/concepts/bioes-tagging.md, wiki/entities/conll-2003.md
   ```

### Query

Ask questions against the wiki. Read `index.md` first, then drill into relevant pages. If the answer is new analysis/comparison, **file it back** into the wiki as a new page (don't let valuable synthesis disappear into chat history).

### Lint

Periodically run:
- Check for contradictions between pages.
- Find stale claims newer sources have superseded.
- Find orphan pages (no inbound links from index.md).
- Suggest sources/papers worth ingesting next.

## Workflow with paper7

[paper7](https://github.com/lucianfialho/paper7) converts arXiv papers to clean markdown.

```bash
# From research/ directory:
paper7 fetch 2407.12345 --output raw/
# paper7 will put a 2407.12345.md file in research/raw/

# Then, in a Claude session:
# "Read research/raw/2407.12345.md and ingest into the wiki."
```

## Editorial voice

- **Be opinionated.** A neutral summary is less useful than "this paper claims X but their setup is contrived; doesn't transfer to BR".
- **Quote sparingly.** Synthesize.
- **Mark uncertainty.** "unclear", "needs verification", "untested in our domain" are valid notes.
- **Link aggressively.** If two pages discuss the same concept under different names, merge or cross-link.
