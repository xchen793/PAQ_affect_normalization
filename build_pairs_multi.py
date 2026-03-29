import json
from pathlib import Path

# ---- CONFIG (edit if needed) ----
IMAGES_ROOT = Path(r"static/images")   # folder that contains bf_happy, bf_sad, ..., wm_sad
REF_SUBDIR  = "ref_face"
OUT_DIR     = Path("config")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON    = OUT_DIR / "endpoints.json"       # main endpoints mapping
OUT_LOG     = OUT_DIR / "endpoints_log.json"   # verbose log (every print stored here)

INTERVALS = [(1,10), (11,20), (21,30), (31,40), (41,50), (51,60), (61, 70), (71, 80), (81, 90), (91,100)]

# ---- logging helper: print + store ----
LOGS = []
def log(level: str, msg: str, **extra):
    text = f"[{level}] {msg}"
    print(text)
    entry = {"level": level, "message": msg}
    if extra:
        entry.update(extra)
    LOGS.append(entry)

def pad3(n: int) -> str:
    return f"{n:03d}"

def interval_for(idx: int):
    for lo, hi in INTERVALS:
        if lo <= idx <= hi:
            return lo, hi
    return None

def scan_category(cat_dir: Path):
    """
    Return mapping idx(str)-> endpoints dict for files in <cat>/ref_face/NNN.png.
    Only include idx where BOTH endpoints exist in ref_face/.
    Also emit log entries for every message (CHECK/WARN/OK/SAVE).
    """
    ref_dir = cat_dir / REF_SUBDIR
    if not ref_dir.exists():
        log("SKIP", f"{cat_dir.name}: no '{REF_SUBDIR}/' folder", category=cat_dir.name, path=str(ref_dir))
        return {}

    # list all PNGs (case-insensitive) in ref_face
    files = [p for p in ref_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    samples = [p.name for p in sorted(files)[:5]]
    log("CHECK",
        f"{cat_dir.name}: found {len(files)} PNG(s) under {ref_dir}",
        category=cat_dir.name, path=str(ref_dir), count=len(files), samples=samples)

    out = {}
    for p in files:
        stem = p.stem  # '001'
        if not stem.isdigit():
            # ignore names that aren't purely digits
            continue
        idx = int(stem)
        if not (1 <= idx <= 100):
            continue

        lo, hi = interval_for(idx)
        if lo is None:
            continue

        lo_path = ref_dir / f"{pad3(lo)}.png"
        hi_path = ref_dir / f"{pad3(hi)}.png"
        missing = []
        if not lo_path.exists(): missing.append(pad3(lo))
        if not hi_path.exists(): missing.append(pad3(hi))

        if missing:
            log("WARN",
                f"{cat_dir.name}: idx {pad3(idx)} missing endpoint(s) {', '.join(missing)}",
                category=cat_dir.name, idx=pad3(idx), interval={"lo": pad3(lo), "hi": pad3(hi)}, missing=missing)
            continue

        left_rel  = f"images/{cat_dir.name}/ref_face/{pad3(lo)}.png"
        right_rel = f"images/{cat_dir.name}/ref_face/{pad3(hi)}.png"
        out[pad3(idx)] = {
            "lo": lo,
            "hi": hi,
            "left":  left_rel,
            "right": right_rel
        }

        # log the saved endpoints
        log("SAVE",
            f"{cat_dir.name}: idx {pad3(idx)} -> endpoints ({pad3(lo)}, {pad3(hi)})",
            category=cat_dir.name, idx=pad3(idx),
            endpoints={"lo": pad3(lo), "hi": pad3(hi), "left": left_rel, "right": right_rel})

    log("OK", f"{cat_dir.name}: {len(out)} indices with valid endpoints",
        category=cat_dir.name, valid_count=len(out))
    return out

def main():
    root = IMAGES_ROOT
    log("INFO", f"Scanning IMAGES_ROOT = {root.resolve()}", images_root=str(root.resolve()))

    if not root.exists():
        log("ERROR", "IMAGES_ROOT does not exist. Fix IMAGES_ROOT at top of script.")
        # still flush logs so you can see them on disk
        with open(OUT_LOG, "w", encoding="utf-8") as f:
            json.dump(LOGS, f, indent=2)
        return

    cats = [d for d in root.iterdir() if d.is_dir()]
    sample_names = [d.name for d in sorted(cats)[:10]]
    log("INFO", f"Found {len(cats)} top-level folders.", count=len(cats), samples=sample_names)

    result = {"categories": [], "map": {}}
    for cat_dir in sorted(cats, key=lambda p: p.name):
        cat_map = scan_category(cat_dir)
        if cat_map:
            result["categories"].append(cat_dir.name)
            result["map"][cat_dir.name] = cat_map

    # write endpoints.json
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    log("DONE",
        f"Wrote {OUT_JSON} (categories with endpoints: {len(result['categories'])})",
        categories=len(result["categories"]))

    # write the log JSON (every print captured)
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        json.dump(LOGS, f, indent=2)

if __name__ == "__main__":
    main()
