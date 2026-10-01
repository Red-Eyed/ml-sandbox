# ML Sandbox

ML Sandbox is a Jupyter-first collection of small experiments that reveal the tensors and
state hidden by high-level deep-learning APIs. Each notebook starts with one concrete question,
implements the mechanism with basic PyTorch tensor operations, changes one variable, and checks
the result.

The notebooks are the project. Python modules are limited to lightweight display tooling; core
ML behavior stays visible in notebook cells where it can be inspected and modified.

## Current Scope

The repository grows one reviewed experiment at a time. Each completed notebook stands alone and
keeps the mechanism being taught visible in its cells rather than moving it into a training
framework.

## Experiments

1. [Where does a parameter gradient live?](01_parameters/parameter_and_gradient.ipynb) inspects
   parameter values, the autograd graph, hand-derived gradients, per-example contributions,
   accumulation across backward passes, and resetting `.grad` to `None`.

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
