# curator.py
# Topic curation for GTA / Virtual TA deployments.
# Loads kb/topics.json (or any path you provide) to define RAG topics.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


DEFAULT_KB_DIR = Path("kb")
DEFAULT_MANIFEST = DEFAULT_KB_DIR / "topics.json"


@dataclass
class CuratorResult:
    knowledge_bases: Dict[str, Path]
    default_topic: str
    warnings: List[str]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_topics(manifest_path: Optional[str] = None) -> CuratorResult:
    """
    Load topic -> file mappings from a JSON manifest.

    Manifest format:
    {
      "default_topic": "Syllabus",
      "topics": [
        {"name": "Syllabus", "file": "Syllabus.txt"},
        {"name": "Diodes", "file": "Diodes.txt"},
        {"name": "Problems", "file": "Problems.txt"}
      ]
    }
    """
    warnings: List[str] = []

    mpath = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not mpath.exists():
        raise FileNotFoundError(f"Missing topic manifest: {mpath}")

    data = _read_json(mpath)

    kb_dir = mpath.parent
    topics = data.get("topics", [])
    default_topic = data.get("default_topic", "")

    knowledge_bases: Dict[str, Path] = {}

    for t in topics:
        name = str(t.get("name", "")).strip()
        rel_file = str(t.get("file", "")).strip()
        if not name or not rel_file:
            warnings.append("Skipping an entry with missing name or file.")
            continue

        fpath = (kb_dir / rel_file).resolve()
        knowledge_bases[name] = fpath

        if not fpath.exists():
            warnings.append(f"File not found for topic '{name}': {fpath}")

    if not knowledge_bases:
        raise ValueError("No valid topics found in the manifest.")

    if not default_topic or default_topic not in knowledge_bases:
        # fall back to first topic
        fallback = next(iter(knowledge_bases.keys()))
        if default_topic and default_topic not in knowledge_bases:
            warnings.append(f"Default topic '{default_topic}' not found; using '{fallback}'.")
        default_topic = fallback

    return CuratorResult(
        knowledge_bases=knowledge_bases,
        default_topic=default_topic,
        warnings=warnings,
    )


def generate_manifest_from_files(
    kb_dir: str = "kb",
    out_path: str = "kb/topics.json",
    default_topic: Optional[str] = None,
    include_exts: Tuple[str, ...] = (".txt", ".pdf"),
) -> Path:
    """
    Create a topics.json by scanning a directory for .txt/.pdf files.

    Topic name defaults to the filename stem:
      kb/Diodes.txt -> "Diodes"
    """
    kb = Path(kb_dir)
    if not kb.exists():
        raise FileNotFoundError(f"KB directory not found: {kb}")

    files = []
    for p in sorted(kb.iterdir()):
        if p.is_file() and p.suffix.lower() in include_exts:
            files.append(p)

    if not files:
        raise ValueError(f"No knowledge files found in: {kb}")

    topics = [{"name": f.stem, "file": f.name} for f in files]

    if default_topic is None:
        default_topic = topics[0]["name"]

    manifest = {
        "default_topic": default_topic,
        "topics": topics,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def _print_warnings(warnings: List[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for w in warnings:
        print(f"- {w}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Curate topics for a GTA/VTA RAG app.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen", help="Generate kb/topics.json from kb directory files.")
    p_gen.add_argument("--kb-dir", default="kb")
    p_gen.add_argument("--out", default="kb/topics.json")
    p_gen.add_argument("--default-topic", default=None)

    p_check = sub.add_parser("check", help="Validate an existing manifest and list topics.")
    p_check.add_argument("--manifest", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "gen":
        out = generate_manifest_from_files(
            kb_dir=args.kb_dir,
            out_path=args.out,
            default_topic=args.default_topic,
        )
        print(f"Wrote: {out}")
        return 0

    if args.cmd == "check":
        res = load_topics(args.manifest)
        _print_warnings(res.warnings)
        print(f"Default topic: {res.default_topic}")
        print("Topics:")
        for k, v in res.knowledge_bases.items():
            print(f"- {k}: {v}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
