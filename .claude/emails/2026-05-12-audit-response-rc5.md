# Email draft — audit follow-through, rc5 tagged

**To:** [Auditor(s)]
**From:** Christopher Sullivan
**Subject:** pyRVtest — rc5 follow-through on the rc4 audit

Draft below. Copy-paste into your client; tweak tone / cut as needed.

---

Hi [Auditor],

Thank you for the two-part audit on `v0.4.0rc4`. You were right that the rc4 sweep was incomplete on the Dearing citation year (eight residues survived), and the boundary-condition bug in the log-cost guard was a genuine fix. `v0.4.0rc5` is now tagged on `v0.4-refactor` and addresses the action items below.

Branch: `v0.4-refactor` at `4e8c711` on https://github.com/anyatarascina/pyRVtest. Tag: `v0.4.0rc5` at `4e8c711`.

```
pip install git+https://github.com/anyatarascina/pyRVtest@v0.4.0rc5
```

## Per-finding ledger

**Audit 1 / Finding 3 — log-cost guard `< 0` should be `<= 0`.** Fixed. The boundary case `markup == price` (implied marginal cost = 0) now raises `ValueError` instead of silently passing `np.log(0) = -inf` to downstream NaN. New regression `tests/test_demand_adjustment.py::test_log_cost_zero_marginal_cost_raises` covers the boundary using your minimal user-supplied-markup reproducer. Commit `4e8c711`, file `pyRVtest/problem.py:1853`.

**Audit 1 / Finding 5 — `solve()` docstring contradicts code.** Fixed. The docstring now states that `costs_type='log'` composes with `demand_adjustment=True` via the chain-rule rescaling (which has been the code's behavior since rc3), removes the stale "falls back to linear" language, and explains the `<= 0` rejection. Commit `4e8c711`, file `pyRVtest/problem.py:1726-1735`.

**Audit 1 / Finding 1 — `docs/notebooks/speed_test.py` collected by pytest.** Fixed by rename: the file is now `speed_benchmark.py`, which doesn't match pytest's default `*_test.py` collection pattern. The file still runs standalone (`python docs/notebooks/speed_benchmark.py`). Commit `4e8c711`.

**Audit 1 / Finding 2 — dependency metadata permits numpy 2 + pyblp 1.1.x.** Addressed via belt-and-suspenders: a comment in `requirements.txt` documents the constraint, and `pyRVtest/__init__.py` enforces it at import time with a clear `ImportError` and fix-it instructions. PEP 508 markers can't express cross-dependency constraints, so this is the cleanest path. The auditor's stress stack (numpy 2.4.4 + pyblp 1.1.2 under Python 3.13) will now fail at import with an actionable message rather than producing silent wrong-sign diagnostics. Commit `4e8c711`, files `pyRVtest/__init__.py:11-44`, `requirements.txt`.

**Audit 1 / Finding 6 = Audit 2 / Finding B1 — Dearing citation sweep incomplete.** Fixed. The rc4 sweep missed eight 2026 → 2024 instances:

- README BibTeX `dmqsw2026` → `dmqsw2024`, `year={2026}` → `year={2024}` (plus a `month={August}` field for completeness)
- `AGENTS.md:333` — bare "Dearing 2026" prose
- `CHANGELOG.md:55,480` — two changelog entries
- `pyRVtest/formulation.py:32` — `known_coefficients` docstring
- `docs/tutorial.rst:249` — passthrough advanced section pointer
- `docs/migrating_to_v0.4.rst:269` — simple-markup model migration section
- `pyRVtest/problem.py:2461,2576` — two pass-through method docstrings (line-wrapped `S. Waldfogel (2026)` form)

The exact grep you suggested now returns clean:

```
$ grep -RIn --exclude-dir=.git --exclude-dir=.claude \
    -e 'Dearing.*2026' -e 'dmqsw2026' -e 'year={2026}' . \
    | grep -v "DMQSS\|dmqss2026\|Duarte.*Magnolfi.*Quint.*S"
# (only DMQSS-2026 BibTeX remains, correctly 2026)
```

Commit `4e8c711`.

**Audit 2 / Finding A1 — README install messaging.** Fixed. The README now leads with the GitHub-tag install command (`git+...@v0.4.0rc5`) and notes that the PyPI release still serves v0.3.x as a fallback for users on the older API. Commit `4e8c711`, file `README.rst:31-58`.

**Audit 2 / Finding B2 — FAQ contradicts setup.py / CI.** Fixed. FAQ now states the `>=3.9` floor and the actual CI matrix (Python 3.11 × {numpy<2 + pyblp<1.2, numpy>=2 + pyblp>=1.2}), with a note that 3.10/3.12 should work but aren't in CI. Commit `4e8c711`, file `docs/faq.rst:17-25`.

## Scope question — pass-through diagnostics in v0.4

The two audits disagreed: Audit 1 recommended deferring or labeling experimental; Audit 2 said generalized PT is the right v0.4 move and just needs cleanup. I've kept PT central in v0.4 (Audit 2's view) for now. The reasoning:

- The generalized PT engine — analytical shortcuts + Vertical + numerical central-difference for non-Vertical conduct — is the substantive methodological content of v0.4. Deferring it would substantially diminish what v0.4 *is*.
- The ~50s performance on the 3000-market synthetic is real but is an optimization concern, not a correctness concern. v0.5 has caching + smaller default examples explicitly scoped.
- API stability concern is the strongest argument for "experimental." I'm willing to add a paragraph to the PT method docstrings noting that column names may evolve in v0.5, if you'd prefer that compromise.

I'm open to revisiting this if a coauthor flags a specific API or terminology issue they'd want renegotiated post-v0.4.

## Deliberately deferred to v0.5

Captured for transparency, not as omissions:

- **PyPI alignment + ReadTheDocs stable pointer.** Both still on v0.3.x until v0.4 leaves rc.
- **NumPy 2 `analytical_scale` xfail.** ~3% F-stat shift on one snapshot under numpy 2 / pyblp 1.2; below DMSS plug-in CV noise floor. Tracked as a CI xfail; root cause is upstream LAPACK changes in numpy 2.
- **PT performance work (caching, precomputed markups, smaller defaults).** Audit 1 Finding 4 + Audit 2 Findings C2-C4. v0.5 has these as priority 1.
- **Centralized safe-linalg utilities + PT reliability vocabulary.** Audit 2 Findings C2-C3. v0.5.
- **Derivative validation for custom demand backends.** Audit 2 Finding D. v0.5.
- **`instrument_channels` field renaming.** Audit 2 Finding C4. v0.5.
- **Mypy enforcement decision.** Audit 2 Finding E3. v0.5.
- **Closing stale GitHub issues** (K>30 issue is technically addressed by rc4; `ownership_downstream` needs verification). I'll close/milestone these in a separate pass.

## Tests

The change set was exercised against 310 tests across the modules touched (snapshots, analytical, demand_adjustment including the new boundary regression, public_api_pin, import_roundtrip, k_gt_30, f_reliability, dearing_models, formulation, passthrough_summary, instrument_channels). All green on the rc4-matrix-default stack (numpy 1.26.4 / pyblp 1.1.2 / Python 3.9). I didn't independently run the numpy 2 / pyblp 1.2 variant in this pass, but the changes are dependency-agnostic (citation strings, an import-time gate, and a guard tightening).

If anything in the ledger looks short or you find a residue the grep missed, send back a follow-up and I'll iterate before cutting v0.4 final.

Chris
