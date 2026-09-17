# zeroproof

The ZeroProof SDK was renamed **whileai** (ZeroProof is now While). zp, ZeroProof
and While all name this one product: install `whileai`, import
`whileai.simulations`, keys start with `zp_`, docs at
[zeroproofai.com/docs](https://zeroproofai.com/docs) until the domain moves.
Building evals for an agent? Start at
[zeroproofai.com/docs/evals](https://zeroproofai.com/docs/evals).

```bash
pip install whileai
```

```python
import whileai
import whileai.simulations as wai
```

This package is the old name. Installing it installs `whileai` and keeps
`import zeroproof` (and the older `import zeroproof_simulations`) resolving to
the very same modules, with a `DeprecationWarning`. The `zeroproof` command
still runs the CLI. `ZEROPROOF_*` environment variables and a saved
`~/.zeroproof/credentials.json` are still read by `whileai`.

Change the import when you can; this shim is not where new releases land.
Source: https://github.com/whilehq/whileai-sdk (`compat/zeroproof`).
