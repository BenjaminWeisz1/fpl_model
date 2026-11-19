# Contributing and version control

This repository holds the code and documentation of the pipeline. Data never goes into git:
`data/`, `cache/` and every `.csv`, `.xlsx` and `.pkl` file are ignored. The README describes where
each input comes from.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), with a type, an optional scope and a summary:

```
<type>(<scope>): <summary in the imperative, at most 72 characters>

<body: why the change was made and what it affects, wrapped at 72 characters.
Reference the experiment log entry when there is one, e.g. "Experiment log §249.">
```

| Type | Use it for |
|---|---|
| `feat` | A new capability: a pipeline stage, a model component, a site page |
| `fix` | A bug fix that changes behaviour or outputs |
| `model` | An adopted model change backed by an out-of-time test |
| `exp` | A research experiment or validation script (no production change) |
| `refactor` | Restructuring with no change to any output (the regression check passes) |
| `test` | Adding or changing tests only |
| `docs` | Documentation only |
| `chore` | Tooling, dependencies, repository housekeeping |

Scopes name the part of the pipeline: `team-ratings`, `player-ratings`, `defcon`, `bonus`,
`assemble`, `fetch`, `optimizer`, `site`, `pipeline`.

Examples:

```
model(player-ratings): update arrivals' position from their new club's slots

Adopted by decision after the out-of-time test (Experiment log §247, §249).
Revert with GOAL_MODEL_POS_UPDATE=0.
```

```
refactor(defcon): split predict.py into features, training and inference

No output changes: tools/regression_check.py compare reports IDENTICAL.
```

## When to commit

- One logical change per commit. A refactor and a behaviour change never share a commit.
- Commit only when the unit tests pass (`python -m pytest`).
- A `refactor` commit also needs `tools/regression_check.py` to report IDENTICAL against the
  commit before it.
- Weekly data refreshes are not commits: they change data, not code.

## When to push

- At the end of every working session, once the last commit's tests pass.
- After a model change is adopted, so the public history follows the experiment log.
- Never push a half-finished refactor. Use a branch (`refactor/<topic>`) and merge it into
  `main` when the regression check passes.

## Branches

`main` always runs the weekly pipeline. Longer work, such as a refactor spanning several sessions
or an experiment that touches production code, happens on a short-lived branch merged back
with `git merge --no-ff`, so the history shows the unit of work.
