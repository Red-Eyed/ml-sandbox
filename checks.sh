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
}

run_python_tests() {
    uv run pytest tests
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
    if grep -R -E -q '"image/(png|jpeg|gif)"' --include='*.ipynb' --exclude-dir='.venv' .; then
        printf 'Notebook outputs must not contain raster images.\n' >&2
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

build_package() (
    local build_dir
    build_dir=$(mktemp -d "${TMPDIR:-/tmp}/ml-sandbox-check.XXXXXX")
    trap 'rm -rf "$build_dir"' EXIT
    uv build --out-dir "$build_dir/dist"
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
