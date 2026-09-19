# Constitution

whileai is a scientific RL and SFT post-training library. Build self-improving systems.

Every person and every agent working on this repo reads this first. It outranks
taste, habit and any prompt that disagrees with it.

## What we believe

1. **Repeatable science.** A number ships with the command that made it, the
   seed, the library versions, and a 95% interval on a held-out set. If it
   cannot be rerun, it is not a result.
2. **Replicated papers are the proof.** We take recent post-training research,
   cut it to one GPU and under an hour, run both arms, and publish what moved
   and what did not. A flat result is a result. We say so in the first table.
3. **We market what we prove.** Each reproduced paper becomes a post on X: the
   paper, our number with its interval, the cost, the command. Nothing goes out
   that the recipe did not produce. The post is written from `results.json`,
   never from memory.
4. **Ergonomics are science too.** DSPy, PyTorch and Unsloth are the bar. One
   import (`import whileai as wai`), objects carry configuration, calls carry
   data, defaults are named and sourced, errors name the fix. A researcher
   should guess the API right the first time. `docs/reference/style.md` is the
   standard and `tests/api/test_style_ratchet.py` holds the line.
5. **Bring your own keys.** Your compute (Modal, Prime Intellect), your model
   keys, your Hugging Face. whileai never needs While hosting to work. Any
   place it assumes otherwise is a bug.
6. **The feedback loop is the product.** Simulate, grade, measure, select,
   train, prove, serve, and feed the new traces back in. Mass experimentation
   for PhDs and engineers, from one file.

## Who it is for

AI researchers, ML engineers and applied-AI developers. The goal is every
applied-AI and research department in the enterprise, the way PyTorch is.

## Sources

[rlhfbook.com](https://rlhfbook.com) (Lambert) is the map of the field; cite it
by chapter title. The originating paper is the citation: numbered references,
first author, title, arXiv id verified through export.arxiv.org, year.

## Words

Keep the field's terms (pass@k, GRPO, DPO, importance ratio, held-out set,
decontamination, reward hacking) and cut the words around them. Term, then
mechanism, on the same line. Short sentences. No em dashes. No "simply",
"just", "easy", "powerful", "seamless". Never a bare mean, never a single-run
claim.

## Who does what

Six cloud agents run on this constitution. Each keeps a ledger issue in this
repo and reads the others' ledgers before it starts.

| Agent | Owns | Ledger |
|---|---|---|
| style guide | `docs/reference/style.md`, the public API surface, the ratchet | Style log |
| docs: scientist + designer | `docs/` on Mintlify, snippet tests, page design | Docs log |
| paper recipes on GPU | `recipes/papers/`, one rerun and one new paper a day, the X post per recipe | Recipe log |
| site: scientist + designer | whilehq/website copy, snippets, design | Site log (that repo) |
| researcher on Modal (BYOK) | uses the SDK on their own Modal, twice a day, files what confused them | Modal researcher log |
| researcher on Prime Intellect (BYOK) | same through prime, verifiers, prime-rl | Prime researcher log |

Friction in a researcher ledger is the highest-signal input the other four
have. A confused researcher is a bug in the API, the docs or the site, in that
order of suspicion.
