# Legal Search MCP

English | [한국어](README.ko.md)

Korean case law, statutes and sentencing data as [MCP](https://modelcontextprotocol.io)
tools. Connect an AI client to find judgments, look up statutes as of a date,
summarise a judgment and calculate statutory and guideline sentencing ranges.

Four read-only tools:

| Tool | What it does |
|---|---|
| `precedent_search` | Find judgments by facts, charge, court, year or case number |
| `precedent_dive` | Read one judgment body and answer a question about it |
| `statute_lookup` | Statute and administrative-rule articles, current or as of a date |
| `sentencing_analysis` | Statutory ranges, guidelines, factors and observed sentences for one charge; calculate ranges from supplied findings |

This repository publishes the MCP tools used by
[로풀 (Lawful)](https://lawful.crow-tit.com), a free Korean legal-AI service.
The web application and hosted Agent API are maintained separately. The
research prototype and benchmark archive live in
[`legal_mcp`](https://github.com/LimEulYoung/legal_mcp) — see [Paper](#paper).

## Quick start — hosted

The hosted service provides the full corpus, including statute amendment
history, with no rate limits. Three ways to connect:

**Claude / ChatGPT on the web — no code.** Add a custom connector with this
URL, sign in with OAuth when asked, and the four tools appear. No API key
needed:

```
https://mcp.crow-tit.com/mcp
```

Step-by-step with the exact menu clicks: [crow-tit.com/docs#mcp](https://crow-tit.com/docs#mcp)

**Claude Code** — one command, with a free key from
[console.crow-tit.com](https://console.crow-tit.com):

```bash
claude mcp add --transport http lawful https://mcp.crow-tit.com/mcp \
  --header "Authorization: Bearer ct_..."
```

**Clients with HTTP MCP configuration** — use the same key. Configuration
formats vary; for clients that accept an `mcpServers` HTTP entry:

```json
{
  "mcpServers": {
    "lawful": {
      "url": "https://mcp.crow-tit.com/mcp",
      "headers": { "Authorization": "Bearer ct_..." }
    }
  }
}
```

## Quick start — self-host

Requires **Python 3.11+**. Install in a virtual environment; the bundled
sample works without API keys:

```bash
git clone https://github.com/LimEulYoung/lawful-mcp
cd lawful-mcp
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

For a local stdio client, replace the path below with the absolute path to
your clone. The client starts the server:

```json
{
  "mcpServers": {
    "lawful-mcp": {
      "command": "/absolute/path/to/lawful-mcp/.venv/bin/lawful-mcp"
    }
  }
}
```

For HTTP at `http://127.0.0.1:8100/mcp`, run:

```bash
.venv/bin/lawful-mcp --transport http --port 8100
```

The self-hosted server has no authentication or usage metering. Access control
for a shared HTTP endpoint belongs to your deployment.

### Configuration and data flow

Settings are listed in [`.env.example`](.env.example). The server reads process
environment variables; it does **not** load `.env` automatically. Export values
in the launching shell or set them in your MCP client's `env` configuration.

With the defaults, three tools search and calculate locally.
`precedent_dive` additionally sends the selected public judgment text, case
metadata and the supplied question to your configured model endpoint. Keep
that question limited to issues in the public judgment.

```bash
export DIVE_API_KEY=...
export DIVE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export DIVE_MODEL=...
```

All three values are required to register `precedent_dive`. Without them,
the other three tools still work. Restart the server after changing settings.

Optional dense search (`USE_DENSE=1`) sends search queries to an embedding
endpoint and, when reranking is enabled, queries and candidate excerpts to a
reranking endpoint. It requires the `dense` extra and corpus vectors, which
the bundled sample does not contain.

### Check the installation

The tests use the bundled sample and constructed test data; no network or
model keys are needed:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

## The corpus

The tools read one SQLite file. The bundled sample is about 48 MiB:

| | Sample (`data/fixture.db`) | Hosted |
|---|---|---|
| Judgments | 700 | 220,000+ |
| Statutes | 27 core laws, current text | All, with amendment history |
| Administrative rules | 20 | All |
| Sentencing guidelines | Complete | Complete |
| Charge taxonomy | Complete | Complete |

The sample supports local trials and testing. Use the hosted service for the
full corpus, or set `CORPUS_DB` to a compatible database you maintain.

Statute lookups distinguish repeal from loss of effect, such as expiry, when
the corpus supplies that information. The sample's lapse register is currently
empty; expiry and reinstatement are covered by separate test data.

Default search combines trigram and [Kiwi](https://github.com/bab2min/kiwipiepy)
morpheme indexes using reciprocal rank fusion. The morpheme index makes
two-character charge names such as `사기`, `절도` and `폭행` searchable.

Judgments are non-copyrightable under Article 7 of the Korean Copyright Act.
Court and case-number provenance is preserved in the data.

### Building your own

[`scripts/build_sample_db.py`](scripts/build_sample_db.py) extracts a sample
from an existing compatible corpus and rebuilds its search indexes. It also
documents the schema; it does not download the hosted corpus:

```bash
.venv/bin/python scripts/build_sample_db.py --source /path/to/corpus.db \
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

- **Lawful Agent API** — one question in, a grounded answer out.
  [crow-tit.com](https://crow-tit.com)
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

Report vulnerabilities privately using [SECURITY.md](SECURITY.md).
Code is [MIT licensed](LICENSE).
