"""
blind_recheck.py  -  a blind second labelling of a random sample of the real photos, to measure how
consistent the labels are (agreement and Cohen's kappa). The photos and their labels are not changed.

STEP 1   python blind_recheck.py make                 (once; refuses to redo it without --force)
   * draws 150 real SesameMulti photos: about the same number per class, within a class in proportion to
     its sources, and at most one photo per near-duplicate group (fixed seed)
   * copies them to outputs/blind_recheck/images/ as img_001 ... img_150, in random order
   * writes outputs/blind_recheck/answer_sheet.xlsx: one row per image and an empty 'your_label' column
     with a drop-down list of the four classes
   * the true labels go to outputs/blind_recheck/_key_do_not_open.csv - do not open it before step 3

STEP 2   python blind_recheck.py label
   * opens a page in your browser that shows the photos one at a time, enlarged, in the order of the sheet
   * press 1-4 (or click) to pick Bacterial Blight, Healthy Leaf, Phyllody or Powdery Mildew; 0 = not sure
     (use it rarely); the left and right arrow keys move between photos, so any answer can be changed
   * every answer is written to answer_sheet.xlsx at once: you can stop (Ctrl+C here) and continue later
   * it never reads the key file, so the true labels are never shown
   * do not edit answer_sheet.xlsx in Excel while this runs; typing in the sheet by hand (codes BB, HL, PI,
     PM also work) is the alternative to this step
   Best done without looking anything up, labelling each photo by what it shows.

STEP 3   python blind_recheck.py score
   * agreement, Cohen's kappa with a 95% bootstrap interval, agreement per class and source, and the
     confusion matrix -> outputs/blind_recheck/results.xlsx and results.txt
"""
import argparse
import datetime as dt
import shutil

import numpy as np
import pandas as pd

from common import GP_ROOT, SPLIT_DIR, OUT_ROOT, CLASSES, CLASS_CODE, Tee

OUT = OUT_ROOT / "blind_recheck"
KEY = OUT / "_key_do_not_open.csv"
SHEET = OUT / "answer_sheet.xlsx"
SEED = 2611
CODE_TO_CLASS = {v: k for k, v in CLASS_CODE.items()}
NOT_SURE = "not sure"
DISPLAY = {"Bacteria Blight Infected": "Bacterial Blight", "Healthy Leaf": "Healthy Leaf",
           "Phyllody Infected": "Phyllody", "Powdery Mildew": "Powdery Mildew"}


def cmd_make(args, log):
    if KEY.exists() and not args.force:
        raise SystemExit(f"{KEY} already exists - the sample is made once. Use --force only to start again.")
    sp = pd.read_csv(SPLIT_DIR / "splits_multi.csv", keep_default_na=False)
    real = sp[~sp.is_synthetic.astype(str).str.lower().isin(["true", "1"])].copy()
    rng = np.random.default_rng(SEED)
    one = (real.sample(frac=1.0, random_state=SEED).drop_duplicates("group_id")   # one photo per group
           .sort_values("uid").reset_index(drop=True))
    per_class = {c: args.n // len(CLASSES) + (1 if i < args.n % len(CLASSES) else 0) for i, c in enumerate(CLASSES)}
    picks = []
    for cls in CLASSES:
        g = one[one.cls == cls]
        share = g.source.value_counts(normalize=True)
        want = (share * per_class[cls]).round().astype(int)
        while want.sum() != per_class[cls]:                  # fix rounding so the class total is exact
            want[share.idxmax()] += 1 if want.sum() < per_class[cls] else -1
        for src, n in want.items():
            pool = g[g.source == src]
            picks.append(pool.iloc[np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))])
    sample = pd.concat(picks).reset_index(drop=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, r in sample.iterrows():
        src = GP_ROOT / r.file.replace("\\", "/")           # the split files may hold Windows paths
        name = f"img_{i + 1:03d}{src.suffix.lower()}"
        shutil.copy2(src, img_dir / name)
        names.append(name)
    sample.insert(0, "image", names)
    sample[["image", "uid", "cls", "source", "group_id", "file"]].to_csv(KEY, index=False, encoding="utf-8")

    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    ws.append(["image", "your_label", "note (optional)"])
    for n in names:
        ws.append([n, None, None])
    dv = DataValidation(type="list", formula1='"' + ",".join(CLASSES) + '"', allow_blank=True)
    dv.error, dv.errorTitle = "Pick one of the four classes (or type BB, HL, PI or PM).", "Unknown class"
    dv.showErrorMessage = False                              # so the short codes can be typed too
    ws.add_data_validation(dv)
    dv.add(f"B2:B{len(names) + 1}")
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width, ws.column_dimensions["C"].width = 14, 28, 40
    wb.save(SHEET)
    log(f"Sample of {len(sample)} photos -> {img_dir}")
    log(sample.groupby(["cls", "source"]).size().to_string())
    log(f"Fill in {SHEET} (do not open {KEY.name}), then run:  python blind_recheck.py score")


LABEL_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Blind re-check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#f6f6f4; --card:#ffffff; --ink:#1f2328; --muted:#5f6368; --line:#d6d6d2; --accent:#2e6db4;
          --chosen-bg:#e5f4e4; --chosen:#1f7a3d; --error:#b42318; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, "Segoe UI", Arial, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:16px; }
  header { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:6px; }
  h1 { font-size:19px; margin:0; }
  #progress { color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
  .help { margin:0 0 10px; color:var(--muted); font-size:14px; }
  .photo { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; text-align:center; }
  .photo img { width:min(100%, 60vh); aspect-ratio:1; object-fit:contain; display:block; margin:0 auto;
               background:#111; border-radius:4px; }
  #name { color:var(--muted); font-size:13px; margin-top:6px; }
  .answers { display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-top:12px; }
  button { font:inherit; color:inherit; border:1px solid var(--line); border-radius:8px; background:var(--card);
           cursor:pointer; padding:12px 8px; }
  button:hover { border-color:var(--accent); }
  button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  button.chosen { border-color:var(--chosen); background:var(--chosen-bg); font-weight:600; }
  .key { display:block; font-size:12px; color:var(--muted); }
  .nav { display:flex; gap:8px; margin-top:8px; }
  .nav button { flex:1; padding:9px 8px; }
  #msg { min-height:22px; margin-top:10px; color:var(--muted); }
  #msg.error { color:var(--error); font-weight:600; }
  #done { display:none; background:var(--card); border:1px solid var(--chosen); border-radius:8px; padding:14px 16px;
          margin-top:12px; }
  code { background:#eeeeea; padding:1px 5px; border-radius:4px; }
  @media (max-width:560px) { .answers { grid-template-columns:repeat(2, 1fr); } }
</style></head>
<body><main>
<header><h1>Blind re-check</h1><span id="progress">Loading...</span></header>
<p class="help">Pick the class each photo shows, without looking anything up. Keys: <b>1-4</b> pick a class,
<b>0</b> = not sure (use it rarely), <b>left / right arrows</b> move between photos. Every answer is saved to
answer_sheet.xlsx straight away.</p>
<div class="photo"><img id="img" alt="Photo to label"><div id="name"></div></div>
<div class="answers" id="answers"></div>
<div class="nav"><button id="back">&larr; Back</button><button id="notsure"><span class="key">0</span>Not sure</button>
<button id="next">Skip &rarr;</button></div>
<div id="msg" role="status"></div>
<div id="done"><b>Every photo has an answer.</b> You can still go back and change any of them. When you are happy,
close this tab, press Ctrl+C in the terminal window, and run <code>python blind_recheck.py score</code></div>
</main>
<script>
let S = null, i = 0, busy = false;
const $ = id => document.getElementById(id);
function firstOpen(from) {
  const n = S.names.length;
  for (let k = 0; k < n; k++) { const j = (from + k) % n; if (!S.answers[S.names[j]]) return j; }
  return -1;
}
function answered() { return S.names.filter(n => S.answers[n]).length; }
function msg(text, error) { $("msg").textContent = text || ""; $("msg").className = error ? "error" : ""; }
function show() {
  const name = S.names[i], cur = S.answers[name] || "";
  $("img").src = "/img/" + encodeURIComponent(name);
  $("name").textContent = name;
  $("progress").textContent = "Photo " + (i + 1) + " of " + S.names.length + " · " + answered() + " answered";
  document.querySelectorAll("#answers button").forEach(b => b.classList.toggle("chosen", b.dataset.value === cur));
  $("notsure").classList.toggle("chosen", cur === S.not_sure);
  $("done").style.display = firstOpen(0) < 0 ? "block" : "none";
  new Image().src = "/img/" + encodeURIComponent(S.names[(i + 1) % S.names.length]);   // preload the next photo
}
async function answer(value) {
  if (busy) return;
  busy = true;
  const name = S.names[i];
  try {
    const r = await fetch("/answer", {method: "POST", headers: {"Content-Type": "application/json"},
                                      body: JSON.stringify({name: name, label: value})});
    const j = await r.json();
    if (!j.ok) { msg(j.error, true); return; }
    S.answers[name] = value;
    msg(j.saved ? "" : j.warning, !j.saved);
    const nxt = firstOpen(i + 1);
    i = nxt >= 0 ? nxt : Math.min(i + 1, S.names.length - 1);
    show();
  } catch (e) {
    msg("The labelling program does not answer. Is it still running in the terminal window?", true);
  } finally { busy = false; }
}
function move(step) { i = (i + step + S.names.length) % S.names.length; msg(""); show(); }
async function load() {
  try { S = await (await fetch("/state")).json(); }
  catch (e) { msg("The labelling program does not answer. Start it with: python blind_recheck.py label", true); return; }
  S.classes.forEach((c, k) => {
    const b = document.createElement("button");
    b.dataset.value = c;
    b.innerHTML = '<span class="key">' + (k + 1) + '</span>' + S.display[k];
    b.onclick = () => answer(c);
    $("answers").appendChild(b);
  });
  $("back").onclick = () => move(-1);
  $("next").onclick = () => move(1);
  $("notsure").onclick = () => answer(S.not_sure);
  const open = firstOpen(0);
  i = open >= 0 ? open : 0;
  show();
}
document.addEventListener("keydown", e => {
  if (!S || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key >= "1" && e.key <= String(S.classes.length)) answer(S.classes[Number(e.key) - 1]);
  else if (e.key === "0") answer(S.not_sure);
  else if (e.key === "ArrowLeft" || e.key === "Backspace") { e.preventDefault(); move(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); move(1); }
});
load();
</script></body></html>
"""


def cmd_label(args, log):
    import json
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote

    from openpyxl import load_workbook

    if not SHEET.exists():
        raise SystemExit(f"{SHEET} not found - run  python blind_recheck.py make  first.")
    wb = load_workbook(SHEET)
    ws = wb["labels"]
    row_of = {str(ws.cell(r, 1).value): r for r in range(2, ws.max_row + 1) if ws.cell(r, 1).value}
    names = list(row_of)
    img_dir = OUT / "images"
    allowed = set(CLASSES) | {NOT_SURE, ""}
    lock = threading.Lock()

    def current(name):
        v = ws.cell(row_of[name], 2).value
        v = "" if v is None else str(v).strip()
        return CODE_TO_CLASS.get(v.upper(), v)          # a code typed in the sheet by hand counts too

    def save():
        try:
            wb.save(SHEET)
            return True
        except PermissionError:                          # Excel has the file open
            return False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):                       # keep the terminal quiet
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=200):
            self.send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, LABEL_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/state":
                with lock:
                    answers = {n: current(n) for n in names}
                self.send_json(dict(names=names, answers=answers, classes=CLASSES,
                                    display=[DISPLAY[c] for c in CLASSES], not_sure=NOT_SURE))
            elif self.path.startswith("/img/"):
                name = unquote(self.path[len("/img/"):])
                if name not in row_of:                   # only the photos listed in the sheet
                    self.send(404, b"not found", "text/plain")
                    return
                path = img_dir / name
                ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                self.send(200, path.read_bytes(), ctype)
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/answer":
                self.send(404, b"not found", "text/plain")
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                name, label = str(data.get("name", "")), str(data.get("label", ""))
            except (ValueError, TypeError):
                self.send_json(dict(ok=False, error="Bad request."), 400)
                return
            if name not in row_of or label not in allowed:
                self.send_json(dict(ok=False, error=f"Unknown photo or label: {name} / {label}"), 400)
                return
            with lock:
                ws.cell(row_of[name], 2).value = label or None
                saved = save()
                n_done = sum(1 for n in names if current(n))
            self.send_json(dict(ok=True, saved=saved, answered=n_done, total=len(names),
                                warning="" if saved else "Not saved yet: answer_sheet.xlsx is open in Excel. Close it "
                                "there; the next answer saves everything."))

    server = None
    for port in range(args.port, args.port + 20):        # the next free port if this one is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port between {args.port} and {args.port + 19}.")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    done = sum(1 for n in names if current(n))
    log(f"Labelling page: {url}  ({done} of {len(names)} photos already answered)")
    log("Keep this window open while you label. Press Ctrl+C here when you are done.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with lock:
            ok = save()
            done = sum(1 for n in names if current(n))
        log(f"{done} of {len(names)} photos answered" + ("" if ok else
            " - NOT SAVED: close answer_sheet.xlsx in Excel and run the label step once more"))
        if done == len(names):
            log("Next:  python blind_recheck.py score")


def kappa(a, b, labels):
    """Cohen's kappa (unweighted), the same as sklearn's cohen_kappa_score(a, b, labels=labels)."""
    idx = {c: k for k, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(1) @ m.sum(0)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_score(args, log):
    key = pd.read_csv(KEY, keep_default_na=False)
    ans = pd.read_excel(SHEET, sheet_name="labels").rename(columns={"your_label": "second"})
    ans["second"] = ans.second.fillna("").astype(str).str.strip()
    ans["second"] = [CODE_TO_CLASS.get(x.upper(), x) for x in ans.second]
    ans["second"] = ["" if x.lower() in (NOT_SURE, "?") else x for x in ans.second]   # not sure = left empty
    unknown = sorted(set(ans.second) - set(CLASSES) - {""})
    if unknown:
        raise SystemExit(f"Unknown labels in the answer sheet: {unknown}. Use the drop-down or BB, HL, PI, PM.")
    d = key.merge(ans[["image", "second"]], on="image", how="left")
    d["second"] = d.second.fillna("")
    blank = d[d.second == ""]
    d = d[d.second != ""].reset_index(drop=True)
    agree = (d.cls == d.second).mean()
    k = kappa(d.cls, d.second, CLASSES)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(d), len(d))
        boots.append(kappa(d.cls.iloc[idx], d.second.iloc[idx], CLASSES))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    cm = pd.crosstab(d.cls.rename("dataset label"), d.second.rename("second label")).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0)
    per = (d.assign(agree=d.cls == d.second).groupby(["cls", "source"])
           .agg(images=("agree", "size"), agreement=("agree", "mean")).reset_index())
    lines = [f"Blind re-check, scored {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"Labelled: {len(d)} of {len(key)} images ({len(blank)} left empty)",
             f"Agreement: {agree:.1%}",
             f"Cohen's kappa: {k:.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f})",
             "", "Confusion matrix (rows = dataset label, columns = second label):", cm.to_string(),
             "", "Agreement per class and source:", per.to_string(index=False)]
    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")
    with pd.ExcelWriter(OUT / "results.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([dict(labelled=len(d), left_empty=len(blank), agreement=agree, cohen_kappa=k,
                           kappa_ci95_low=lo, kappa_ci95_high=hi)]).to_excel(xw, sheet_name="summary", index=False)
        cm.to_excel(xw, sheet_name="confusion")
        per.to_excel(xw, sheet_name="per_class_source", index=False)
        d[d.cls != d.second].to_excel(xw, sheet_name="disagreements", index=False)
    for line in lines:
        log(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "label", "score"])
    ap.add_argument("--n", type=int, default=150, help="make: number of photos")
    ap.add_argument("--force", action="store_true", help="make: draw a new sample (deletes nothing else)")
    ap.add_argument("--port", type=int, default=8765, help="label: local port for the labelling page")
    ap.add_argument("--no-browser", action="store_true", help="label: do not open the browser by itself")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = Tee(OUT / "blind_recheck_log.txt")
    log(f"=== blind_recheck.py {args.step}  {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    if args.step == "make":
        cmd_make(args, log)
    elif args.step == "label":
        cmd_label(args, log)
    else:
        cmd_score(args, log)


if __name__ == "__main__":
    main()
