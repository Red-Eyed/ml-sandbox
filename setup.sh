#!/usr/bin/env bash

set -euo pipefail

require_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    printf 'uv is required: https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
}

project_directory() {
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd
}

install_project() {
    uv sync --locked
}

verify_editable_install() {
    uv run python -c 'import ml_sandbox; print(f"Editable package: {ml_sandbox.__file__}")'
}

main() {
    require_uv

    local project_dir
    project_dir="$(project_directory)"
    cd "$project_dir"

    install_project
    verify_editable_install
}

main "$@"
