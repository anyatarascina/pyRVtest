"""Conduct model library: class-based ConductModel API (Option B).

Placeholder for v0.4 step 5 (mechanical models: Bertrand, Cournot,
Monopoly, PerfectCompetition, MixCournotBertrand, PartialCollusion),
step 12 (simple-markup models after Dearing verification:
ConstantMarkup, RuleOfThumb, CostPlus), and step 14 (labor-side models:
Monopsony, BertrandWages, CournotEmployment, NashBargaining).

See `.claude/plans/v0.4-refactor.md` §4.2 for the full API design.
Backward compat: `ModelFormulation(model_downstream='bertrand', ...)`
is preserved as a deprecation alias constructing the right class.
"""

__all__: list[str] = []
