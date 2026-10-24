#!/usr/bin/env bash

set -euo pipefail

readonly MAXIMUM_NOTEBOOK_BYTES=$((512 * 1024))

project_directory() {
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd
}

notebook_files() {
    find . -path './.venv' -prune -o -type f -name '*.ipynb' -print | sort
}

run_shell_checks() {
    bash -n setup.sh checks.sh
}

run_static_checks() {
    uv run ruff check .
    uv run ruff format --check .
    uv run pyrefly check src
    uv run pyrefly check --preset default \
        src/ml_sandbox/numerical_finiteness \
        src/ml_sandbox/matcher_lab \
        06_hungarian_matcher/execute.py \
        06_hungarian_matcher/matcher_lab.ipynb \
        05_numerical_stability/numerical_stability.ipynb
}

run_python_tests() {
    uv run pytest tests src/ml_sandbox/numerical_finiteness src/ml_sandbox/matcher_lab
}

run_notebook_checks() {
    local notebooks=()
    local notebook

    while IFS= read -r notebook; do
        notebooks+=("$notebook")
    done < <(notebook_files)

    if ((${#notebooks[@]} == 0)); then
        return
    fi

    uv run pytest --nbmake "${notebooks[@]}"
}

check_notebook_sizes() {
    local notebook_count=0
    local notebook
    local notebook_bytes

    while IFS= read -r notebook; do
        notebook_count=$((notebook_count + 1))
        notebook_bytes=$(wc -c < "$notebook")
        if ((notebook_bytes > MAXIMUM_NOTEBOOK_BYTES)); then
            printf '%s is %d bytes; the limit is %d bytes.\n' \
                "$notebook" "$notebook_bytes" "$MAXIMUM_NOTEBOOK_BYTES" >&2
            exit 1
        fi
    done < <(notebook_files)

    printf '%d' "$notebook_count"
}

check_notebook_outputs() {
    if grep -R -E -q '"image/(png|gif)"' --include='*.ipynb' --exclude-dir='.venv' .; then
        printf 'Raster notebook outputs must use compact JPEG images.\n' >&2
        exit 1
    fi

    if grep -R -q '<script' --include='*.ipynb' --exclude-dir='.venv' .; then
        printf 'Notebook outputs must not depend on executable JavaScript.\n' >&2
        exit 1
    fi
}

check_image_assets() {
    local image_file
    image_file=$(find . -path './.venv' -prune -o -type f \
        \( -iname '*.gif' -o -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) \
        -print -quit)
    if [[ -n "$image_file" ]]; then
        printf 'Image assets are not allowed: %s\n' "$image_file" >&2
        exit 1
    fi
}

verify_wheel() {
    local distribution_directory="$1"
    uv run python - "$distribution_directory" <<'PY'
"""Check wheel contents and execute the public API from the wheel itself."""

import sys
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import Tensor

wheels = list(Path(sys.argv[1]).glob("*.whl"))
assert len(wheels) == 1
wheel = wheels[0]
with ZipFile(wheel) as archive:
    names = archive.namelist()
assert "ml_sandbox/py.typed" in names
assert "ml_sandbox/numerical_finiteness/_check.py" in names
assert "ml_sandbox/matcher_lab/geometry.py" in names
assert "ml_sandbox/matcher_lab/assignment.py" in names
assert not any(Path(name).name.startswith("test_") for name in names)

sys.path.insert(0, str(wheel))
import ml_sandbox.numerical_finiteness as finiteness

assert finiteness.__file__ and finiteness.__file__.startswith(str(wheel))


def loss(x: Tensor) -> Tensor:
    """Retain a finite forward value with an infinite derivative at zero."""
    return x.sqrt()


def sample(generator: torch.Generator) -> finiteness.Sample:
    """Supply a named finite boundary input to the packaged checker."""
    return finiteness.Sample((torch.tensor(0.0, requires_grad=True),), "exact_zero")


report = finiteness.check_finiteness(
    loss, sample, max_trials=3, required_gradients=(0,), max_examples=1, quiet=True
)
assert report.trials == report.random.nonfinite_gradient == 3
assert report.random.nonfinite_loss == 0
assert report.strategies["exact_zero"].rate("nonfinite_gradient") == 1.0
assert len(report.examples) == 1
import ml_sandbox.matcher_lab as matcher_lab

assert matcher_lab.__file__ and matcher_lab.__file__.startswith(str(wheel))
box = matcher_lab.Box("G1", 0, 0, 4, 2)
prediction = matcher_lab.Prediction(matcher_lab.Box("P1", 0, 0, 4, 2), 0.9)
evaluation = matcher_lab.evaluate_costs((prediction,), (box,), matcher_lab.baseline())
assert matcher_lab.solve_assignment(evaluation.total).pairs == ((0, 0),)
print(f"Wheel APIs verified: {wheel.name}")
PY
}

build_package() (
    local build_dir
    build_dir=$(mktemp -d "${TMPDIR:-/tmp}/ml-sandbox-check.XXXXXX")
    trap 'rm -rf "$build_dir"' EXIT
    uv build --out-dir "$build_dir/dist"
    verify_wheel "$build_dir/dist"
)

main() {
    local project_dir
    local notebook_count

    project_dir="$(project_directory)"
    cd "$project_dir"

    ./setup.sh
    run_shell_checks
    run_static_checks
    run_python_tests
    run_notebook_checks
    notebook_count="$(check_notebook_sizes)"
    check_notebook_outputs
    check_image_assets
    build_package

    printf 'All checks passed. Validated %d notebook(s).\n' "$notebook_count"
}

main "$@"
