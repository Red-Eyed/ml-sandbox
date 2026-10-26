"""Verify interactive initialization without saving runtime widget outputs."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.notebooknode import NotebookNode


def read_notebook(path: Path) -> NotebookNode:
    """Narrow nbformat's legacy multi-version return type at the library boundary."""
    match nbformat.read(path, as_version=4):
        case NotebookNode() as notebook:
            return notebook
        case _:
            raise ValueError("Expected a version 4 notebook record")


def main() -> None:
    """Execute the editor and atomically save its source without runtime outputs.

    Live widgets remain available when running in Jupyter. Execution or size
    validation failure leaves the existing notebook untouched.
    """
    path = Path(__file__).with_name("matcher_lab.ipynb")
    notebook = read_notebook(path)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3", store_widget_state=False)
    client.execute(cwd=str(path.parent.parent))
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type == "code":
            cell.outputs = []
    serialized = nbformat.writes(notebook)
    if len(serialized.encode()) >= 512 * 1024:
        raise ValueError("Notebook exceeds the 512 KiB limit")
    temporary = path.with_suffix(".ipynb.tmp")
    temporary.write_text(serialized)
    temporary.replace(path)
    print(f"Executed {path.name}: interactive editor verified; no saved runtime outputs")


if __name__ == "__main__":
    main()
