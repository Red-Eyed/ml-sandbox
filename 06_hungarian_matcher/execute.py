"""Execute the lab and save static SVG evidence without serialized live widget output."""

from pathlib import Path
from xml.etree import ElementTree

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
    """Execute initialization, reject nonstatic evidence, and atomically replace the notebook.

    Live outputs tagged remove-output are discarded after successful execution.
    The running Jupyter notebook still displays those widgets normally. Execution
    or artifact-validation failure leaves the existing notebook untouched.
    """
    path = Path(__file__).with_name("matcher_lab.ipynb")
    notebook = read_notebook(path)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3", store_widget_state=False)
    client.execute(cwd=str(path.parent.parent))
    notebook.metadata.pop("widgets", None)
    svg_count = 0
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
        if "remove-output" in cell.metadata.get("tags", []):
            cell.outputs = []
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValueError("Notebook contains an execution error")
            for mime, value in output.get("data", {}).items():
                if mime not in {"text/plain", "text/html", "image/svg+xml"}:
                    raise ValueError(f"Unsupported saved MIME type: {mime}")
                if mime == "image/svg+xml":
                    svg_count += 1
                    root = ElementTree.fromstring(value)
                    if any(
                        element.tag.rsplit("}", 1)[-1] in {"image", "script"}
                        for element in root.iter()
                    ):
                        raise ValueError(
                            "SVG must contain vector content without scripts or images"
                        )
    if svg_count == 0:
        raise ValueError("The lab must retain a static SVG preview")
    serialized = nbformat.writes(notebook)
    if len(serialized.encode()) >= 512 * 1024:
        raise ValueError("Notebook exceeds the 512 KiB limit")
    temporary = path.with_suffix(".ipynb.tmp")
    temporary.write_text(serialized)
    temporary.replace(path)
    print(f"Executed {path.name}: {svg_count} static SVGs; no saved widget state")


if __name__ == "__main__":
    main()
