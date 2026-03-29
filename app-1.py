from __future__ import annotations
import os, re, json, tempfile, hashlib, random
from uuid import uuid4
from datetime import datetime
from collections import defaultdict
from flask import Flask, jsonify, render_template, url_for, abort, request, redirect, session

# =========================
# App & constants
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

ALL_IDENTITIES = ["mm", "wm", "bf", "wf", "mf", "bm", "if", "im"]
# Fixed 4-identity groups (user will take exactly one of these)
GROUP_A = ["bm", "if", "wm", "mf"]
GROUP_B = ["bf", "im", "wf", "mm"]

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_num = re.compile(r"(\d+)")

# Files & dirs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_ROOT = os.path.join(BASE_DIR, "static", "images")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SUBMIT_DIR = os.path.join(BASE_DIR, "submissions")
PAIRWISE_DIR = os.path.join(SUBMIT_DIR, "pairwise")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(PAIRWISE_DIR, exist_ok=True)

# Optional config artifacts
ENDPOINTS_LOG_PATH  = os.path.join(CONFIG_DIR, "endpoints_log.json")
ENDPOINTS_JSON_PATH = os.path.join(CONFIG_DIR, "endpoints.json")

# Stage parameters
MAX_REFS_PER_FOLDER = 2                # We want exactly 2 left references per folder if available
DELTA = 10                              # Baseline gap for pre-PAQ pairwise (right = left + 10)

# =========================
# Utilities
# =========================
def natural_key(s): return [int(t) if t.isdigit() else t.lower() for t in _num.split(s)]
def iso_utc_now():  return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        try: os.remove(tmp)
        except Exception: pass

def read_json_if_exists(path):
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _file_exists(rel_path: str) -> bool:
    abs_path = os.path.join(app.static_folder, rel_path.replace("/", os.sep))
    return os.path.isfile(abs_path)

# =========================
# Filesystem helpers (images)
# =========================
def list_folders():
    if not os.path.isdir(STATIC_IMAGES_ROOT): return []
    return sorted(
        [d for d in os.listdir(STATIC_IMAGES_ROOT)
         if os.path.isdir(os.path.join(STATIC_IMAGES_ROOT, d)) and not d.startswith(".")],
        key=natural_key
    )

def list_slider_frames(folder):
    base = os.path.join(STATIC_IMAGES_ROOT, folder)
    if not os.path.isdir(base): return []
    names = []
    for n in os.listdir(base):
        full = os.path.join(base, n)
        if os.path.isdir(full) and n == "ref_face":
            continue
        if os.path.splitext(n)[1].lower() in ALLOWED_EXTS:
            names.append(n)
    names.sort(key=natural_key)
    return names

def list_slider_frames_manifest(folder):
    return [{"name": n, "url": url_for('static', filename=f"images/{folder}/{n}")}
            for n in list_slider_frames(folder)]

def resolve_ref_face_url(folder: str, ref_idx: int | str):
    """
    Return (url, name) for the exact reference face number found in ref_face/.
    Accepts 33 or "033" etc. Tries:
      images/<folder>/ref_face/033.png
      images/<folder>/ref_face/33.png
    Fallbacks:
      images/<folder>/033.png
      the smallest numeric frame in images/<folder>/
    """
    try:
        k = int(ref_idx)
    except Exception:
        return None, None

    ref_face_dir = f"images/{folder}/ref_face"
    cand_padded = f"{ref_face_dir}/{k:03d}.png"
    cand_plain  = f"{ref_face_dir}/{k}.png"
    if _file_exists(cand_padded):
        return url_for("static", filename=cand_padded), f"{k:03d}.png"
    if _file_exists(cand_plain):
        return url_for("static", filename=cand_plain), f"{k}.png"

    root_png = f"images/{folder}/{k:03d}.png"
    if _file_exists(root_png):
        return url_for("static", filename=root_png), f"{k:03d}.png"

    folder_dir = os.path.join(app.static_folder, "images", folder)
    if os.path.isdir(folder_dir):
        frames = sorted(
            int(n[:3]) for n in os.listdir(folder_dir)
            if n.endswith(".png") and len(n) >= 7 and n[:3].isdigit()
        )
        if frames:
            f = frames[0]
            return url_for("static", filename=f"images/{folder}/{f:03d}.png"), f"{f:03d}.png"
    return None, None

def _list_pngs(folder: str) -> list[int]:
    root = os.path.join(app.static_folder, "images", folder)
    if not os.path.isdir(root): return []
    ints = []
    for n in os.listdir(root):
        if n.endswith(".png") and len(n) >= 7 and n[:3].isdigit():
            try: ints.append(int(n[:3]))
            except: pass
    return sorted(ints)

def scan_ref_left_indices_per_folder(folder: str) -> list[int]:
    """
    Return numeric indices available in images/<folder>/ref_face/*.{png|jpg|...},
    accepting filenames like 33.png or 033.png. Sorted unique list.
    """
    ref_dir = os.path.join(STATIC_IMAGES_ROOT, folder, "ref_face")
    if not os.path.isdir(ref_dir): return []
    lefts = []
    for fn in os.listdir(ref_dir):
        root, ext = os.path.splitext(fn)
        if ext.lower() not in ALLOWED_EXTS:
            continue
        m = _num.search(root)
        if m:
            try:
                idx = int(m.group(1))
                if 0 <= idx <= 99:
                    lefts.append(idx)
            except Exception:
                pass
    return sorted(set(lefts))

# =========================
# Session + storage helpers
# =========================
def _ensure_sid():
    '''
    Ensure a session id exists.
    If Prolific ID was already set, NEVER overwrite it.
    '''
    sid = session.get("sid")
    if not sid:
        sid = uuid4().hex
        session["sid"] = sid
    return sid

@app.route("/submit_prolific_id", methods=["POST"])
def submit_prolific_id():
    prolific_id = (request.form.get("prolific_id") or "").strip()

    if not prolific_id:
        abort(400, "Missing Prolific ID")

    # # Optional sanity check (recommended)
    # if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", prolific_id):
    #     abort(400, "Invalid Prolific ID format")

    # 🔑 Prolific ID becomes THE session id
    session.clear()                 # prevent leakage from prior sessions
    session["sid"] = prolific_id
    session["provenance"] = "prolific"

    return redirect(url_for("introduction"))


def session_file_path(session_id: str) -> str:
    safe = "".join(c for c in (session_id or "") if c.isalnum() or c in "-_")[:64] or "anon"
    return os.path.join(SUBMIT_DIR, f"{safe}.json")

def has_original_answer(session_id: str, folder: str, ref_idx: int) -> bool:
    merged = read_json_if_exists(session_file_path(session_id)) or {}
    by_folder = (merged.get("answers") or {}).get(folder) or {}
    return str(ref_idx) in (by_folder.get("responses") or {})

def has_followup_answer(session_id: str, folder: str, ref_idx: int) -> bool:
    merged = read_json_if_exists(session_file_path(session_id)) or {}
    by_folder = (merged.get("answers") or {}).get(folder) or {}
    return f"{ref_idx}f" in (by_folder.get("responses") or {})

def followup_ref_from_original_record(session_id: str, folder: str, ref_idx: int):
    merged = read_json_if_exists(session_file_path(session_id)) or {}
    by_folder = (merged.get("answers") or {}).get(folder) or {}
    rec = (by_folder.get("responses") or {}).get(str(ref_idx))
    if isinstance(rec, dict):
        resp = rec.get("response") or {}
        if resp.get("url") and resp.get("name"):
            return resp["url"], resp["name"]
    return resolve_ref_face_url(folder, ref_idx)

# =========================
# Group assignment & common selection
# =========================
def _seed_from_sid(sid: str) -> int:
    return int(hashlib.sha1(sid.encode()).hexdigest(), 16) % (2**32)

def _assign_id_subset_for_sid(sid: str, force: str | None = None):
    """
    Assign Group A or B deterministically from sid (override with ?group=A|B or FORCE_GROUP).
    """
    if force is None:
        force = (request.args.get("group") or os.environ.get("FORCE_GROUP") or "").upper()
    if force in {"A", "B"}:
        chosen = GROUP_A if force == "A" else GROUP_B
    else:
        h = int(hashlib.sha1(sid.encode()).hexdigest(), 16)
        chosen = GROUP_A if (h % 2 == 0) else GROUP_B
    session["id_subset"] = chosen
    return chosen

def _identity_from_folder(folder: str) -> str:
    # e.g., "mm_happy" -> "mm"
    return folder.split("_", 1)[0]

def _folders_for_group(ids: list[str]) -> list[str]:
    """
    Return folders for both directions (happy/sad) that physically exist:
      <id>_happy and <id>_sad
    """
    out = []
    for ident in ids:
        for dirn in ("happy", "sad"):
            candidate = f"{ident}_{dirn}"
            abs_path = os.path.join(STATIC_IMAGES_ROOT, candidate)
            if os.path.isdir(abs_path):
                out.append(candidate)
    return sorted(out, key=natural_key)

# =========================
# Pairwise-first (baseline Δ=10 using curated lefts)
# =========================
def build_pairwise_first_order(sid: str):
    """
    Baseline pairwise built from curated ref_face/ indices.

    ORDERING:
      - Trials are grouped IDENTITY-BY-IDENTITY.
      - Identities (4 of them) are shown in a RANDOM order.
      - Within each identity, its trials (happy/sad × up to 2 refs) are RANDOMLY ordered.
    """
    rng = random.Random(_seed_from_sid(sid))
    chosen_ids = session.get("id_subset") or _assign_id_subset_for_sid(sid)

    # All folders for this 4-identity subset
    folders = _folders_for_group(chosen_ids)

    # Group folders by identity, e.g. {"mm": ["mm_happy", "mm_sad"], ...}
    folders_by_id = defaultdict(list)
    for f in folders:
        ident = _identity_from_folder(f)
        if ident in chosen_ids:
            folders_by_id[ident].append(f)

    # For each identity, collect its trials
    trials_by_id: dict[str, list[dict]] = defaultdict(list)
    for ident, f_list in folders_by_id.items():
        for f in f_list:
            left_indices = scan_ref_left_indices_per_folder(f)
            # choose up to 2 per folder
            left_indices = left_indices[:MAX_REFS_PER_FOLDER]
            for L in left_indices:
                R = min(99, L + DELTA)

                # exact left file from ref_face/
                ref_dir_rel = f"images/{f}/ref_face"
                left_rel_plain  = f"{ref_dir_rel}/{L}.png"
                left_rel_padded = f"{ref_dir_rel}/{L:03d}.png"
                if _file_exists(left_rel_plain):
                    left_url = url_for("static", filename=left_rel_plain)
                    left_name = f"{L}.png"
                elif _file_exists(left_rel_padded):
                    left_url = url_for("static", filename=left_rel_padded)
                    left_name = f"{L:03d}.png"
                else:
                    left_url, left_name = resolve_ref_face_url(f, L)
                    if not left_url:
                        continue

                right_rel = f"images/{f}/{R:03d}.png"
                if not _file_exists(right_rel):
                    # try a small fallback if needed
                    if R > 0 and _file_exists(f"images/{f}/{R-1:03d}.png"):
                        R -= 1
                        right_rel = f"images/{f}/{R:03d}.png"
                    else:
                        continue

                trials_by_id[ident].append({
                    "identity": ident,
                    "folder": f,
                    "ref_idx": int(L),                   # numeric frame index, e.g., 33
                    "left_url": left_url,
                    "right_url": url_for("static", filename=right_rel),
                    "left_name": left_name,
                    "right_name": f"{R:03d}.png",
                    "flag": "Δ=10 baseline",
                    "jnd": "Δ=10",
                })

    # Randomize order of trials WITHIN each identity
    for ident in trials_by_id:
        rng.shuffle(trials_by_id[ident])

    # Randomize the order of identities themselves
    identities_order = [ident for ident in chosen_ids if ident in trials_by_id]
    rng.shuffle(identities_order)

    # Concatenate: identity by identity
    final_order = []
    for ident in identities_order:
        final_order.extend(trials_by_id[ident])

    app.logger.info(
        f"[PAIRWISE-FIRST Δ=10] built {len(final_order)} pairs "
        f"(subset={session.get('id_subset')}, ids_order={identities_order})"
    )
    return final_order

# Build the PAQ originals *from the same left references as pairwise*, but shuffled
def build_original_order_from_pairwise(sid: str) -> list[tuple[str,int]]:
    """
    Returns a list of (folder, ref_idx) sourced from the current pairwise-first order.
    If the pairwise order isn't in session, we rebuild it (same logic), then shuffle a copy.
    (Shuffling here is ONLY for the PAQ order; baseline pairwise order itself is untouched.)
    """
    rng = random.Random(_seed_from_sid(sid) ^ 0xA5A5A5A5)
    order = session.get("pw_order_baseline") or session.get("pw_order_stage2")
    if not order:
        order = build_pairwise_first_order(sid)
    # Keep only (folder, ref_idx)
    pairs = [(t["folder"], int(t["ref_idx"])) for t in order]
    rng.shuffle(pairs)
    return pairs

# =========================
# Stage-2 order (from PAQ originals/answers)
# =========================
def _parse_frame_number_from_name(name_or_url: str) -> int | None:
    if not name_or_url: return None
    base = os.path.basename(name_or_url.split("?")[0])
    root, _ = os.path.splitext(base)
    if root.isdigit():
        k = int(root)
        if 1 <= k <= 100: return k
    m = _num.search(root)
    if m:
        k = int(m.group(1))
        if 1 <= k <= 100: return k
    return None

def build_stage2_order_from_paq(session_id: str):
    """
    Stage-2 uses:
      - LEFT  = same (folder, ref_idx) and SAME ORDER as pairwise_early baseline
      - RIGHT = user's chosen JND from PAQ original for that (folder, ref_idx)

    We also keep identity info so we can insert ask_difficulty pages identity-by-identity.
    """
    merged = read_json_if_exists(session_file_path(session_id)) or {}
    answers = merged.get("answers") or {}

    # Use the stored baseline order to keep identity-by-identity structure
    baseline_order = session.get("pw_order_baseline")
    if not baseline_order:
        baseline_order = build_pairwise_first_order(session_id)
        session["pw_order_baseline"] = baseline_order

    out = []
    for t in baseline_order:
        folder = t["folder"]
        ref_idx = int(t["ref_idx"])
        ident = t.get("identity") or _identity_from_folder(folder)

        rec = ((answers.get(folder) or {}).get("responses") or {}).get(str(ref_idx), {})
        chosen = None
        if isinstance(rec, dict):
            resp = rec.get("response") or {}
            nm = resp.get("name")
            if isinstance(nm, str) and len(nm) >= 7 and nm[:3].isdigit():
                chosen = int(nm[:3])
            else:
                chosen = _parse_frame_number_from_name(resp.get("url") or resp.get("name"))
        if chosen is None:
            # skip if PAQ original missing for this ref
            continue

        left_url, left_name = resolve_ref_face_url(folder, ref_idx)
        if not left_url:
            continue

        right_url = url_for("static", filename=f"images/{folder}/{int(chosen):03d}.png")
        out.append({
            "identity": ident,
            "folder": folder,
            "ref_idx": int(ref_idx),
            "left_url": left_url,
            "right_url": right_url,
            "left_name": left_name or f"{int(ref_idx):03d}.png",
            "right_name": f"{int(chosen):03d}.png",
            "flag": "PAQ",
            "jnd": "1 JND",
        })

    app.logger.info(
        f"[STAGE2] built {len(out)} pairs from PAQ originals for sid={session_id} "
        f"(preserving baseline pairwise order)"
    )
    return out

# =========================
# ROUTES — High-level flow
# =========================
@app.route("/")
def root():
    return render_template("prolific_id.html")

@app.route("/introduction")
def introduction():
    _ensure_sid()
    return render_template("introduction.html")

@app.route("/pretest")
def pretest():
    _ensure_sid()
    return render_template("pretest.html")

# Continue to pairwise-first
@app.route("/pretest/continue", methods=["GET","POST"])
def pretest_continue():
    sid = _ensure_sid()
    return redirect(url_for("pairwise_intro", session=sid))

# Pairwise intro → pairwise trials (pairwise-first)
@app.route("/pairwise/intro")
def pairwise_intro():
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    if "id_subset" not in session:
        _assign_id_subset_for_sid(sid)

    # rebuild for clean start
    session.pop("pw_order_stage2", None)
    session.pop("pw_order_baseline", None)

    order = build_pairwise_first_order(sid)
    if not order:
        abort(500, "Could not build pairwise trials (no images found).")

    # keep a copy specifically for “pairwise lefts” reference
    session["pw_order_stage2"] = order
    session["pw_order_baseline"] = order
    return redirect(url_for("pairwise_trial", i=0, session=sid))

@app.route("/pairwise/trial")
def pairwise_trial():
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    try:
        i = int(request.args.get("i", "0"))
    except Exception:
        i = 0

    order = session.get("pw_order_stage2") or build_pairwise_first_order(sid)
    if not order:
        abort(404, "No pairwise order available.")
    total = len(order)
    if i < 0 or i >= total:
        abort(404, "Pairwise trial index out of range.")

    t = order[i]
    app.logger.info(
        f"[PAIRWISE] trial render: i={i}/{total}, "
        f"identity={t.get('identity')}, folder={t['folder']}, "
        f"left={t['left_url']}, right={t['right_url']}"
    )
    return render_template(
        "pairwise_stage2.html",
        session_id=sid,
        trial_idx=i,
        total=total,
        folder=t["folder"],
        flag=t.get("flag", "Δ=10 baseline"),
        jnd=t.get("jnd", "Δ=10"),
        left_url=t["left_url"],
        right_url=t["right_url"],
        left_name=t.get("left_name", ""),
        right_name=t.get("right_name", ""),
        api_submit_url=url_for("pairwise_submit"),
        end_url=url_for("pairwise_done", session=sid),
        is_last=(i == total - 1),
        back_url=url_for("pairwise_intro", session=sid),
    )

@app.post("/pairwise/api/submit")
def pairwise_submit():
    """
    Submit a single pairwise (early) trial.
    After the LAST trial for a given identity, we redirect to ask_difficulty.html
    for that identity, then continue to the next identity (or finish).
    """
    payload = request.get_json(force=True) or {}
    sid = (payload.get("sessionId") or session.get("sid") or _ensure_sid())
    session["sid"] = sid

    try:
        i = int(payload.get("trial_idx", payload.get("trialIndex", 0)))
    except Exception:
        i = 0

    order = session.get("pw_order_stage2") or []
    if not order or i < 0 or i >= len(order):
        return jsonify({"ok": False, "error": "Invalid trial index"}), 400

    pw_path = os.path.join(PAIRWISE_DIR, f"{sid}.json")
    store = read_json_if_exists(pw_path) or {}
    if sid not in store:
        store[sid] = []
    store[sid].append({
        "block": "pairwise_early",
        "trial_idx": i,
        "choice": payload.get("choice"),
        "key": payload.get("key"),
        "rt_ms": payload.get("rt_ms"),
        "folder": payload.get("folder"),
        "left_name": payload.get("left_name"),
        "right_name": payload.get("right_name"),
        "submittedAt": iso_utc_now(),
    })
    atomic_write_json(pw_path, store)

    # --- Check if this was the last trial for its identity ---
    cur_trial = order[i]
    cur_identity = cur_trial.get("identity") or _identity_from_folder(cur_trial["folder"])

    # last index for this identity
    last_idx_for_ident = max(
        j for j, t in enumerate(order)
        if (t.get("identity") or _identity_from_folder(t["folder"])) == cur_identity
    )

    if i == last_idx_for_ident:
        # Finished all trials for this identity → go to *early* difficulty page
        next_i = -1
        for j in range(i + 1, len(order)):
            j_ident = order[j].get("identity") or _identity_from_folder(order[j]["folder"])
            if j_ident != cur_identity:
                next_i = j
                break

        ask_url = url_for(
            "pairwise_ask_difficulty",
            session=sid,
            identity=cur_identity,
            next_i=next_i,
        )
        return jsonify({"ok": True, "nextUrl": ask_url})

    # Otherwise, continue to next trial
    next_i = i + 1
    if next_i >= len(order):
        return jsonify({"ok": True, "nextUrl": url_for("pairwise_done", session=sid)})
    return jsonify({"ok": True, "nextUrl": url_for("pairwise_trial", i=next_i, session=sid)})

@app.route("/pairwise/ask_difficulty")
def pairwise_ask_difficulty():
    """
    Page shown RIGHT AFTER finishing all pairwise trials for an identity
    in the early (Δ=10 baseline) block.
    """
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    identity = (request.args.get("identity") or "").strip()
    try:
        next_i = int(request.args.get("next_i", "-1"))
    except Exception:
        next_i = -1

    if not identity:
        abort(400, "Missing identity for difficulty question.")

    return render_template(
        "ask_difficulty.html",
        session_id=sid,
        identity=identity,
        next_i=next_i,
        submit_url=url_for("pairwise_ask_difficulty_submit"),
        block_label="First comparison block",
    )

@app.post("/pairwise/ask_difficulty/submit")
def pairwise_ask_difficulty_submit():
    """
    Handle submission from ask_difficulty.html for the early pairwise block.
    Stores one row per identity under block="pairwise_difficulty".
    """
    sid = (request.form.get("session_id") or session.get("sid") or _ensure_sid())
    session["sid"] = sid

    identity   = (request.form.get("identity") or "").strip()
    difficulty = (request.form.get("difficulty") or "").strip()
    comments   = (request.form.get("comments") or "").strip()
    try:
        next_i = int(request.form.get("next_i", "-1"))
    except Exception:
        next_i = -1

    if not identity:
        abort(400, "Missing identity in difficulty submission.")

    pw_path = os.path.join(PAIRWISE_DIR, f"{sid}.json")
    store = read_json_if_exists(pw_path) or {}
    if sid not in store:
        store[sid] = []

    store[sid].append({
        "block": "pairwise_difficulty",
        "identity": identity,
        "difficulty": difficulty,
        "comments": comments,
        "submittedAt": iso_utc_now(),
    })
    atomic_write_json(pw_path, store)

    # Decide where to go next
    if next_i >= 0:
        next_url = url_for("pairwise_trial", i=next_i, session=sid)
    else:
        next_url = url_for("pairwise_done", session=sid)

    return redirect(next_url)

@app.route("/pairwise/done")
def pairwise_done():
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    return redirect(url_for("paq_index", session=sid))

# =========== PAQ (Stage-1) uses the SAME left references as pairwise, but shuffled ===========
@app.route("/paq/index")
def paq_index():
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    if "id_subset" not in session:
        _assign_id_subset_for_sid(sid)

    # Build PAQ originals directly from pairwise lefts (shuffled)
    pairs = build_original_order_from_pairwise(sid)
    ordered = []
    for qnum, (folder, ref_idx) in enumerate(pairs, start=1):
        done = has_original_answer(sid, folder, ref_idx) and has_followup_answer(sid, folder, ref_idx)
        ordered.append({"qnum": qnum, "folder": folder, "ref_idx": ref_idx, "done": done})

    return render_template("index.html",
                           ordered_questions=ordered,
                           session_id=sid)

@app.route("/<path:folder>/<int:ref_idx>/")
def legacy_slider_redirect(folder, ref_idx):
    session_q = request.args.get("session", "")
    fu = request.args.get("fu", "0")
    return redirect(url_for("paq_slider_page", folder=folder.strip("/"), ref_idx=ref_idx, fu=fu, session=session_q))

@app.route("/paq/<folder>/<int:ref_idx>/")
def paq_slider_page(folder, ref_idx):
    """
    Render PAQ slider for a specific (folder, ref_idx). ref_idx is the ACTUAL numeric frame
    (e.g., 33), not a 0..N-1 index. We just need to resolve the ref image; no 0..N-1 bound check.
    """
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    fu = int(request.args.get("fu", 0))

    if fu == 1:
        ref_url, ref_name = followup_ref_from_original_record(sid, folder, ref_idx)
    else:
        ref_url, ref_name = resolve_ref_face_url(folder, ref_idx)
    if not ref_url:
        app.logger.warning(f"[PAQ] Missing ref image for {folder} #{ref_idx}")
        return redirect(url_for("paq_index", session=sid))

    frames = list_slider_frames_manifest(folder)
    right_url = frames[0]["url"] if frames else url_for("static", filename=f"images/{folder}/001.png")
    last_frame_idx = 0

    return render_template(
        "slider.html",
        session_id=sid,
        folder=folder,
        ref_idx=ref_idx,
        fu=fu,
        ref_url=ref_url,
        right_url=right_url,
        frames=frames,
        last_frame_idx=last_frame_idx,
        index_url=url_for("paq_index", session=sid),
    )

# unified submit alias
@app.post("/api/submit")
def api_submit_compat():
    return paq_api_submit()

@app.post("/paq/api/submit")
def paq_api_submit():
    """
    Save PAQ response. When all originals+follow-ups are done, proceed to Stage-2.
    """
    payload = request.get_json(force=True) or {}

    session_id = (request.args.get("session") or session.get("sid") or _ensure_sid()).strip()
    folder = (request.args.get("folder") or "").strip()
    if not folder:
        return jsonify({"ok": False, "error": "Missing folder"}), 400
    try:
        fu = int(request.args.get("fu", "0"))
    except Exception:
        fu = 0
    try:
        ref_idx = int(request.args.get("ref_idx"))
    except Exception:
        return jsonify({"ok": False, "error": "Missing ref_idx"}), 400

    # Resolve the left image (validation by existence)
    if fu == 1:
        ref_url, ref_name = followup_ref_from_original_record(session_id, folder, ref_idx)
    else:
        ref_url, ref_name = resolve_ref_face_url(folder, ref_idx)
    if not ref_url:
        return jsonify({"ok": False, "error": f"Reference image not found for {folder} #{ref_idx}"}), 400
    reference = {"folder": folder, "name": ref_name, "url": ref_url}

    frames = list_slider_frames_manifest(folder)
    if not frames:
        return jsonify({"ok": False, "error": "No slider frames found in this folder."}), 400

    sfi = None
    if "selectedFrameIndex" in payload:
        try: sfi = int(payload["selectedFrameIndex"])
        except Exception: sfi = None
    if sfi is None and "selectedFrame" in payload:
        try: sfi = int(payload["selectedFrame"]) - 1
        except Exception: sfi = None
    sfi = 0 if sfi is None else max(0, min(sfi, len(frames) - 1))

    response = {
        "index": int(sfi),
        "name": frames[sfi]["name"],
        "url":  frames[sfi]["url"],
    }

    merged_path = session_file_path(session_id)
    merged = read_json_if_exists(merged_path) or {"sessionId": session_id, "answers": {}}
    by_folder = merged["answers"].setdefault(folder, {"responses": {}, "last_ref_idx": None})
    key = str(ref_idx) if fu == 0 else f"{ref_idx}f"
    by_folder["responses"][key] = {
        "reference": reference,
        "response": response,
        "fu": int(fu),
        "submittedAt": iso_utc_now(),
    }
    if fu == 0:
        by_folder["last_ref_idx"] = int(ref_idx)
    atomic_write_json(merged_path, merged)

    # simple plan: originals then follow-ups for the SAME pool
    originals = build_original_order_from_pairwise(session_id)
    plan = [{"folder": f, "ref_idx": i, "fu": 0} for (f,i) in originals] + \
           [{"folder": f, "ref_idx": i, "fu": 1} for (f,i) in originals]

    # find current item and pick next unfinished
    def _is_done(item):
        return has_original_answer(session_id, item["folder"], item["ref_idx"]) if item["fu"]==0 \
               else has_followup_answer(session_id, item["folder"], item["ref_idx"])

    try:
        cur_ix = next(j for j,it in enumerate(plan)
                      if it["folder"]==folder and it["ref_idx"]==ref_idx and int(it["fu"])==int(fu))
    except StopIteration:
        cur_ix = -1

    next_ref = None
    if plan:
        for step in range(1, len(plan)+1):
            ix = (cur_ix + step) % len(plan) if cur_ix >= 0 else (step-1)
            if not _is_done(plan[ix]):
                nxt = plan[ix]
                next_ref = {"folder": nxt["folder"], "ref_idx": nxt["ref_idx"], "fu": int(nxt["fu"])}
                break

    if next_ref:
        next_url = url_for("paq_slider_page",
                           folder=next_ref["folder"], ref_idx=next_ref["ref_idx"],
                           fu=next_ref["fu"], session=session_id)
        completed_all = False
    else:
        next_url = url_for("stage2_intro", session=session_id)
        completed_all = True

    return jsonify({
        "ok": True,
        "next_url": next_url,
        "folder": folder,
        "ref_idx": int(ref_idx),
        "fu": int(fu),
        "completed_all": completed_all
    })

# =========================
# Stage-2: pairwise from PAQ JNDs
# =========================
@app.route("/stage2/intro")
def stage2_intro():
    sid = (request.args.get("session") or session.get("sid") or uuid4().hex[:8])
    session["sid"] = sid
    if "id_subset" not in session:
        _assign_id_subset_for_sid(sid)

    session.pop("pw_order_stage2", None)
    order = build_stage2_order_from_paq(sid)
    if not order:
        return redirect(url_for("paq_index", session=sid))

    session["pw_order_stage2"] = order
    return render_template(
        "stage2_intro.html",
        session_id=sid,
        total=len(order),
        left_img=url_for("static", filename="images/stage2_left.png"),
        right_img=url_for("static", filename="images/stage2_right.png"),
    )

@app.route("/stage2/start")
def stage2_start():
    sid = (request.args.get("session") or session.get("sid") or uuid4().hex[:8])
    session["sid"] = sid

    order = session.get("pw_order_stage2")
    if not order:
        order = build_stage2_order_from_paq(sid)
        if not order:
            return redirect(url_for("paq_index", session=sid))
        session["pw_order_stage2"] = order

    return redirect(url_for("stage2_trial", i=0, session=sid))

@app.route("/stage2/trial")
def stage2_trial():
    sid = (request.args.get("session") or session.get("sid") or "anon")
    session["sid"] = sid
    try:
        i = int(request.args.get("i", "0"))
    except Exception:
        i = 0

    order = session.get("pw_order_stage2") or build_stage2_order_from_paq(sid)
    if not order:
        return redirect(url_for("paq_index", session=sid))

    total = len(order)
    if i < 0 or i >= total:
        return redirect(url_for("end_page"))

    t = order[i]
    return render_template(
        "pairwise_stage2.html",
        session_id=sid,
        trial_idx=i,
        total=total,
        folder=t["folder"],
        flag=t.get("flag", "PAQ"),
        jnd=t.get("jnd", "1 JND"),
        left_url=t["left_url"],
        right_url=t["right_url"],
        left_name=t.get("left_name", ""),
        right_name=t.get("right_name", ""),
        api_submit_url=url_for("stage2_submit"),
        end_url=url_for("end_page"),
        is_last=(i == total - 1),
    )

@app.route("/stage2/ask_difficulty")
def stage2_ask_difficulty():
    """
    Page shown RIGHT AFTER finishing all Stage-2 pairwise trials for an identity.
    """
    sid = (request.args.get("session") or session.get("sid") or _ensure_sid())
    session["sid"] = sid
    identity = (request.args.get("identity") or "").strip()
    try:
        next_i = int(request.args.get("next_i", "-1"))
    except Exception:
        next_i = -1

    if not identity:
        abort(400, "Missing identity for stage2 difficulty question.")

    return render_template(
        "ask_difficulty.html",
        session_id=sid,
        identity=identity,
        next_i=next_i,
        submit_url=url_for("stage2_ask_difficulty_submit"),
        block_label="Second comparison block",
    )

@app.post("/stage2/ask_difficulty/submit")
def stage2_ask_difficulty_submit():
    """
    Handle submission from ask_difficulty.html for the Stage-2 block.
    Stores one row per identity under block="stage2_difficulty".
    """
    sid = (request.form.get("session_id") or session.get("sid") or _ensure_sid())
    session["sid"] = sid

    identity   = (request.form.get("identity") or "").strip()
    difficulty = (request.form.get("difficulty") or "").strip()
    comments   = (request.form.get("comments") or "").strip()
    try:
        next_i = int(request.form.get("next_i", "-1"))
    except Exception:
        next_i = -1

    if not identity:
        abort(400, "Missing identity in stage2 difficulty submission.")

    pw_path = os.path.join(PAIRWISE_DIR, f"{sid}.json")
    store = read_json_if_exists(pw_path) or {}
    if sid not in store:
        store[sid] = []

    store[sid].append({
        "block": "stage2_difficulty",
        "identity": identity,
        "difficulty": difficulty,
        "comments": comments,
        "submittedAt": iso_utc_now(),
    })
    atomic_write_json(pw_path, store)

    # Decide where to go next
    if next_i >= 0:
        next_url = url_for("stage2_trial", i=next_i, session=sid)
    else:
        next_url = url_for("end_page")

    return redirect(next_url)

@app.post("/stage2/api/submit")
def stage2_submit():
    """
    Stage2 pairwise submit:
    - Record the trial.
    - If this was the last trial for a given identity, go to stage2_ask_difficulty.
    - Otherwise continue to the next trial.
    """
    payload = request.get_json(force=True) or {}
    sid = (payload.get("sessionId") or session.get("sid") or "anon")
    session["sid"] = sid
    try:
        i = int(payload.get("trial_idx", payload.get("trialIndex", 0)))
    except Exception:
        i = 0

    order = session.get("pw_order_stage2") or build_stage2_order_from_paq(sid)
    total = len(order)
    if not order or i < 0 or i >= total:
        return jsonify({"ok": False, "error": "Invalid trial index"}), 400

    os.makedirs(PAIRWISE_DIR, exist_ok=True)
    pw_path = os.path.join(PAIRWISE_DIR, f"{sid}.json")
    store = read_json_if_exists(pw_path) or {}
    if sid not in store:
        store[sid] = []
    store[sid].append({
        "block": "stage2_from_paq",
        "trial_idx": i,
        "total": total,
        "choice": payload.get("choice"),
        "key": payload.get("key"),
        "rt_ms": payload.get("rt_ms"),
        "folder": payload.get("folder"),
        "left_name": payload.get("left_name"),
        "right_name": payload.get("right_name"),
        "submittedAt": iso_utc_now(),
    })
    atomic_write_json(pw_path, store)

    # ---- NEW: identity-level difficulty page for Stage2 ----
    cur_trial = order[i]
    cur_identity = cur_trial.get("identity") or _identity_from_folder(cur_trial["folder"])

    last_idx_for_ident = max(
        j for j, t in enumerate(order)
        if (t.get("identity") or _identity_from_folder(t["folder"])) == cur_identity
    )

    if i == last_idx_for_ident:
        # End of this identity's Stage2 trials
        next_i = -1
        for j in range(i + 1, len(order)):
            j_ident = order[j].get("identity") or _identity_from_folder(order[j]["folder"])
            if j_ident != cur_identity:
                next_i = j
                break

        ask_url = url_for(
            "stage2_ask_difficulty",
            session=sid,
            identity=cur_identity,
            next_i=next_i,
        )
        return jsonify({"ok": True, "next_url": ask_url})

    # Otherwise continue as usual
    if i + 1 < total:
        next_url = url_for("stage2_trial", i=i+1, session=sid)
    else:
        next_url = url_for("end_page")

    return jsonify({"ok": True, "next_url": next_url})

# =========================
# End page
# =========================
@app.route("/end")
def end_page():
    sid = session.get("sid") or ""

    merged = read_json_if_exists(session_file_path(sid)) or {}
    answers = merged.get("answers") or {}

    # Count PAQ originals completed
    originals = build_original_order_from_pairwise(sid)
    completed_paq_originals = 0
    for (f, ref_idx) in originals:
        if str(ref_idx) in ((answers.get(f) or {}).get("responses") or {}):
            completed_paq_originals += 1

    # Pairwise/Stage2 summary
    pw_path = os.path.join(PAIRWISE_DIR, f"{sid}.json")
    pw_data = read_json_if_exists(pw_path) or {}
    pw_rows = pw_data.get(sid, []) if isinstance(pw_data, dict) else []

    pairwise_early_trials = sum(1 for r in pw_rows if r.get("block") == "pairwise_early")
    stage2_trials         = sum(1 for r in pw_rows if r.get("block") == "stage2_from_paq")

    return render_template(
        "end.html",
        session_id=sid,
        total_paq_originals=len(originals),
        completed_paq_originals=completed_paq_originals,
        pairwise_early_trials=pairwise_early_trials,
        stage2_trials=stage2_trials,
        api_feedback_url= url_for("api_dummy_feedback") if "api_dummy_feedback" in app.view_functions else ""
    )

# =========================
# Dev server
# =========================
if __name__ == "__main__":
    app.logger.info("Server starting. If you see 404 /favicon.ico, add static/favicon.ico or a <link rel='icon'>.")
    app.run(host="0.0.0.0", port=5000, debug=True)
