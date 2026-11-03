"""Write analysis result files."""

import json
from pathlib import Path


def write_shot_metrics_json(output_path: Path, video_record: dict[str, object]) -> None:
    """Write one video's metrics record as strict, human-readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            video_record,
            output_file,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        output_file.write("\n")
