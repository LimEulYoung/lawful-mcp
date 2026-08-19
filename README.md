# Legal Search MCP

English | [한국어](README.ko.md)

Korean case law, statutes and sentencing data as [MCP](https://modelcontextprotocol.io)
tools. Point an AI client at it and the model can look up statute text as it
stood on a given date, find judgments by the facts of a case, read one
judgment and answer a question about it, and compute a sentencing range the
way a Korean court does.

Five read-only tools:

| Tool | What it does |
|---|---|
| `precedent_search` | Find judgments by facts, charge, court, year or case number |
| `precedent_dive` | Read one judgment body and answer a question about it |
| `statute_lookup` | Statute and administrative-rule articles, current or as of a date |
| `sentence_statistics` | Observed first-instance sentencing distribution for a charge |
| `compute_sentencing_range` | Statutory range → processed range → guideline range → verification |

This is the production system behind [로풀 (Lawful)](https://lawful.crow-tit.com),
a free Korean legal-AI service, and the descendant of the research prototype in
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp) — see [Paper](#paper).

## Quick start — hosted

The full corpus (220k+ judgments, statutes with their amendment history,
administrative rules, sentencing guidelines) is served for you. Get a free
key at [console.crow-tit.com](https://console.crow-tit.com) and add:

```json
{
  "mcpServers": {
    "legal-search": {
      "type": "http",
      "url": "https://mcp.crow-tit.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_KEY" }
    }
  }
}
```

Works with Claude Desktop, Claude Code, Cursor and any other MCP client. No
rate limits.

## Quick start — self-host

The repository ships a sample corpus so a clone runs immediately:

```bash
git clone https://github.com/LimEulYoung/legal-search-mcp
cd legal-search-mcp
pip install -e .
legal-search-mcp                 # stdio
legal-search-mcp --transport http --port 8100
```

Register it with a client:

```json
{
  "mcpServers": {
    "legal-search": {
      "command": "legal-search-mcp"
    }
  }
}
```

Configuration is all environment variables — see [`.env.example`](.env.example).
Nothing is required; the defaults use the bundled corpus.

### The dive tool needs a model

Four of the five tools are pure database reads. `precedent_dive` sends one
public judgment body to a language model and asks it to extract an answer,
so it needs an endpoint:

```bash
export DIVE_API_KEY=...
export DIVE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export DIVE_MODEL=...
```

Without those three the tool is not registered and the other four run
normally.

## The corpus

The tools read one SQLite file. What ships here is a bounded sample, not the
whole thing:

| | Sample (`data/fixture.db`) | Hosted |
|---|---|---|
| Judgments | 800 | 220,000+ |
| Statutes | 27 core laws, current text | All, with amendment history |
| Administrative rules | 20 | All |
| Sentencing guidelines | Complete | Complete |
| Charge taxonomy | Complete | Complete |

The sample is enough to exercise every tool and run the tests. For real work,
use the hosted corpus.

Search combines two lexical indexes: a character-trigram FTS and a
morpheme FTS built with [Kiwi](https://github.com/bab2min/kiwipiepy), fused
with reciprocal rank fusion. The morpheme index is not an optimisation —
Korean charge names are often two characters (`사기`, `절도`, `폭행`), and a
trigram index cannot form a trigram from two characters, so those queries
return nothing without it.

Judgments are non-copyrightable under Article 7 of the Korean Copyright Act.
Court and case-number provenance is preserved in the data.

### Building your own

`scripts/build_sample_db.py` carves a sample out of a full corpus and is also
the reference for the schema — which tables the tools read, how the
full-text indexes are built, and how statute versions resolve:

```bash
python scripts/build_sample_db.py --source /path/to/corpus.db \
    --dest my_sample.db --cases 5000 --statutes all
```

## Paper

The retrieval and tool-use design was evaluated on the Korean Bar Examination:

> **Agentic RAG for Legal Question Answering in Civil Law: Evidence From the Korean Bar Examination**
> Eul Young Lim and Jihun Park
> *IEEE Access*, vol. 14, pp. 124441–124458, 2026.
> [doi:10.1109/ACCESS.2026.3722717](https://doi.org/10.1109/ACCESS.2026.3722717) — open access

Benchmark code, questions and per-model results are archived in
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp). If you use this tool
in research, please cite the paper ([`CITATION.cff`](CITATION.cff)).

## Also available

- **Legal Search API** — one question in, a grounded answer out, in Anthropic
  Messages format. [crow-tit.com](https://crow-tit.com)
- **로풀 (Lawful)** — the free consumer service, 법률AI: chat, search, labour
  calculators, legal document drafting.
  [lawful.crow-tit.com](https://lawful.crow-tit.com)

## Notes

This is a search tool over public legal sources, not legal advice. What it
returns is source material and statistics; deciding what they mean for a
particular matter is a lawyer's job.

Development happens in a private repository and lands here in batches, so
issues are welcome but pull requests may be merged by hand rather than
through the button.

MIT licensed.
