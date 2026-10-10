# Repository Guidelines

## Purpose and Current Scope

This is a Jupyter-first educational repository for making deep-learning training mechanics visible and memorable. Human understanding is the primary design criterion: readers should see the real tensors, change one condition, and build intuition from the result.

The repository currently contains no experiments. Treat it as an intentional notebook-ready scaffold. When work resumes, add one focused notebook for the requested question; do not recreate removed experiments unless explicitly asked.

## Repository Boundaries

Notebooks are the product. Keep the demonstrated machine-learning mechanism directly in notebook cells; never hide it in helpers. Use basic PyTorch tensors, autograd, and `nn.Parameter`, with high-level PyTorch modules only as numerical references.

`src/ml_sandbox/` is an installable package for presentation tooling only. Altair builds charts, Rich presents exact values, and `vl-convert-python` renders charts as inline SVG. `setup.sh` installs the locked project editably, so notebook imports resolve without relative paths. Keep Bash scripts decomposed into short, single-purpose functions.

## Development Commands

Use `uv` for every Python operation:

```bash
./setup.sh                        # Sync the lockfile and install editably
uv run jupyter lab                # Open the interactive notebook environment
./checks.sh                       # Run every repository check
```

`checks.sh` validates Bash syntax, Ruff formatting and linting, Pyrefly types, packaging, notebook size, and visualization-output policy. It executes every notebook it discovers and also supports the intentional zero-notebook state.

## Notebook Design

Keep each notebook small enough to understand in one sitting. Follow this progression: concrete question, minimal implementation, state inspection, one controlled experiment, visualization, and PyTorch verification. Use small visible shapes, explicit tensor names, fixed random seeds, real values, and executable assertions.

Before implementing a notebook, first determine the visualization schema that best explains its question and experiment. Propose that schema with its rationale, discuss it with the user, and wait for explicit approval before writing the notebook. If implementation reveals that the approved visual story needs to change materially, pause and repeat this review rather than choosing a replacement silently.

Visualization carries the explanation. Design the visual story before writing the surrounding prose and code. Use charts, trajectories, annotated tensor diagrams, storyboards, or small multiples as the primary teaching medium. A Rich table is secondary evidence for exact values only when those values add something the visual cannot communicate; never make a sequence of raw tables the notebook's main narrative. Commit executed outputs as compact inline SVG so static GitHub previews work. Each notebook must remain below 512 KiB and contain no raster output, generated image assets, or JavaScript-dependent charts.

Every committed notebook and visualization must be fully compatible with GitHub's static renderer. A reader must see the complete explanation directly on GitHub without executing the notebook, opening an external site, trusting JavaScript, or loading a Jupyter widget. Use static HTML and inline SVG outputs; represent multi-step processes with static trajectories, selected-step storyboards, or small multiples rather than runtime animation.

Before presenting a notebook as complete, inspect its executed output MIME types and rendered visuals. Confirm that the explanation uses only static text or HTML and inline SVG, with no raster output, executable JavaScript, or widget state.

Avoid trainers, registries, configuration systems, base model classes, and one-use abstractions. Prefer readable repetition when extraction would hide the lesson.

## Reviews and Commits

Run `./checks.sh` after every change. Pull requests should name the question being taught and explain how the experiment and visualization answer it. Commit messages use an imperative subject of at most 72 characters, concrete change bullets, and a final paragraph explaining the motivation and design choice.

Do not create commits unless the user explicitly requests a commit.
