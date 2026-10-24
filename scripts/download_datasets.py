from pathlib import Path

from ntv_ai.data.download import main


if __name__ == "__main__":
    raise SystemExit(main(project_root=Path(__file__).resolve().parents[1]))
