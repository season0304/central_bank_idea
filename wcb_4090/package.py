"""Package the new 4090 pipeline, its processed datasets and the local encoder."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wcb-4090.tar.gz")
    args = parser.parse_args()
    included = ["wcb_4090", "pyproject.toml", "models/gte-modernbert-base"]
    included += [f"data/processed_4090/wcb_{task}_5768" for task in ("time", "certain", "stance")]
    paths = []
    for name in included:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run all three preparation commands first")
        for p in ([path] if path.is_file() else sorted(path.rglob("*"))):
            if p.is_file() and not {"__pycache__", ".cache"} & set(p.parts):
                paths.append(p)
    manifest = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        data = json.dumps({"files": manifest}, indent=2).encode()
        member = tarfile.TarInfo("bundle_manifest.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    temporary.replace(args.output)
    print(json.dumps({"path": str(args.output), "bytes": args.output.stat().st_size,
                      "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
