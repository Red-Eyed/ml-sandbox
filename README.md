# ML Sandbox

ML Sandbox is a Jupyter-first collection of small experiments that reveal the tensors and
state hidden by high-level deep-learning APIs. Each notebook starts with one concrete question,
implements the mechanism with basic PyTorch tensor operations, changes one variable, and checks
the result.

The notebooks are the project. Python modules are limited to lightweight display tooling; core
ML behavior stays visible in notebook cells where it can be inspected and modified.

## Current Scope

The repository is currently an empty, notebook-ready scaffold. The next experiment should be one
small notebook built around one explicit question. Add later topics as independent experiments
rather than building a training framework.

## Run the Experiments

Install the locked environment and the project in editable mode, then launch JupyterLab:

```bash
./setup.sh
uv run jupyter lab
```

Run the complete validation suite:

```bash
./checks.sh
```

The checks lint and type-check the package, execute every notebook currently present, enforce the
notebook-size and no-image policies, and verify that the installable package builds. They also pass
when the repository intentionally contains no notebooks.

Altair renders charts as compact inline SVG that remains visible in static GitHub previews. Rich
renders exact tensor values and tables as HTML-capable notebook output. The repository contains no
generated image files or JavaScript-dependent chart outputs.
