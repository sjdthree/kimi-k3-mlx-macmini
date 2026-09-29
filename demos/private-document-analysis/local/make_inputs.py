"""Build the local run's chat inputs from the case PDFs.

The cloud run did not record how the PDFs reached the model (see
../evaluation/methodology.md), so the local run fixes it here and commits the result:
the PDF text from `pdftotext -layout` (which keeps the tables' columns), with trailing
spaces removed and blank-line runs collapsed, then a blank line and the question from
../prompts/. One user message, no system prompt.

    python demos/private-document-analysis/local/make_inputs.py

Needs `pdftotext` from poppler (`brew install poppler`; built with 26.07.0). The JSON
files are committed, so the local run itself does not need it.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent
CASES = {
    "semiconductor": "01_semiconductor_confidential_excursion.pdf",
    "medical_ai": "02_medical_ai_confidential_incident.pdf",
    "rd_tradesecret": "03_rd_tradesecret_invention_strategy.pdf",
}


def pdf_text(pdf: Path) -> str:
    raw = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                         text=True, check=True).stdout
    lines = [line.rstrip() for line in raw.replace("\f", "\n").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def main() -> int:
    out = DEMO / "local" / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    for name, pdf in CASES.items():
        question = (DEMO / "prompts" / f"{name}.txt").read_text().strip()
        content = pdf_text(DEMO / "cases" / pdf) + "\n\n" + question
        path = out / f"{name}.json"
        path.write_text(json.dumps([{"role": "user", "content": content}],
                                   ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {path.relative_to(DEMO)} ({len(content)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
