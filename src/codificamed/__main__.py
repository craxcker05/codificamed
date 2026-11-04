"""CLI: python -m codificamed "nota clínica"."""

from __future__ import annotations

import sys

from codificamed.pipeline import analizar


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Uso: python -m codificamed "nota clínica"')
    res = analizar(" ".join(sys.argv[1:]))
    print(res.model_dump_json(indent=2, exclude={"sugerencias": {"__all__": {"candidatos"}}}))


if __name__ == "__main__":
    main()
