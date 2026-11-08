"""Bundle only code, processed WCB data, and the original local model for Vast.ai."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/wcb-vast.tar.gz")
    args = parser.parse_args()
    include = ["README.md", "pyproject.toml", "wcb_triplet", "scripts", "tests",
               "data/processed/wcb_time_5768", "models/gte-modernbert-base"]
    paths = []
    for name in include:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else path.rglob("*")
        paths.extend(p for p in candidates if p.is_file() and not {"__pycache__", ".cache"} & set(p.parts))
    if not (ROOT / "models/gte-modernbert-base/model.safetensors").is_file():
        raise FileNotFoundError("Download the original GTE model before packaging")
    files = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    manifest = json.dumps({"files": files}, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo("bundle_manifest.json")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    temporary.replace(args.output)
    print(json.dumps({"archive": str(args.output), "files": len(files),
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
