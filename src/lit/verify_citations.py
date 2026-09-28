"""
Automated citation checks (check 1 of the lit-review protocol).

  1. bib vs source: each bib entry's title must appear in the first pages of
     its local PDF (an independent check on the registry metadata).
  2. ledger locators: for every "(PDF pN)" locator, all of that locator's
     keywords must appear on page N of the source PDF. [VISUAL] and TBD
     locators (image-only scans, books without PDF) are listed for manual check.
  3. cross-references: ledger bib_keys exist in the bib; bib entries exist in
     sources.csv; any \\cite{} keys in the deck exist in the bib.

Text is compared after folding ligatures and dropping spaces and hyphens,
since PDF extraction routinely splits or joins words. Exits 1 on failure.

Usage: python3 src/lit/verify_citations.py   (needs pypdf)
"""
import csv
import html
import re
import sys
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[2]
LIT = ROOT / "docs" / "lit_review"
PDFS = ROOT / "Lit Review"
DECK = ROOT / "docs" / "presentation" / "presi.tex"
LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_pages = {}


def squash(s):
    for a, b in LIG.items():
        s = s.replace(a, b)
    s = re.sub(r"\{\\[^}]*\}|\\[a-z]+|[{}\\]", "", s)   # drop TeX markup from bib titles
    return re.sub(r"[\s\-\u2010-\u2015]", "", s).lower()   # spaces, hyphens, en/em dashes


def pages(fname):
    if fname not in _pages:
        _pages[fname] = [squash(p.extract_text() or "") for p in pypdf.PdfReader(PDFS / fname).pages]
    return _pages[fname]


def main():
    fails, manual = [], []
    sources = {r["bib_key"]: r for r in csv.DictReader(open(LIT / "sources.csv")) if r["bib_key"]}
    bib = dict(re.findall(r"@\w+\{(\w+),(.*?)\n\}", (LIT / "references.bib").read_text(), re.S))

    # 1. bib title vs the PDF's own first pages
    for key, body in bib.items():
        if key not in sources:
            fails.append("bib entry %s has no sources.csv row" % key)
            continue
        f = sources[key]["file"]
        if not f:
            manual.append("%s: no local PDF (%s)" % (key, sources[key]["status"]))
            continue
        raw = re.search(r"^\s*title = \{(.*)\}", body, re.M).group(1)   # anchored: not 'booktitle'
        first = pages(f)[:5]
        if sum(map(len, first)) < 200:
            manual.append("%s: image-only PDF, title checked visually only" % key)
        elif "no text title" in sources[key]["notes"] + sources[key]["version"] or "images or blank" in sources[key]["notes"]:
            manual.append("%s: PDF has no text title page - metadata from registry only" % key)
        elif not any(squash(raw)[:40] in p for p in first):
            words = [squash(w) for w in re.findall(r"[A-Za-z]{4,}", raw)]
            hit = sum(any(w in p for p in first) for w in words) / max(len(words), 1)
            if hit >= 0.75:
                manual.append("%s: title matched %.0f%% of words only (OCR scan?) - confirm visually" % (key, 100 * hit))
            else:
                fails.append("%s: bib title not found on first pages of %s" % (key, f))

    # 2. ledger locators: keywords must be on the cited page
    n_checked = 0
    for r in csv.DictReader(open(LIT / "citation_ledger.csv")):
        locs, kws = r["locator"].split(" | "), r["keywords"].split(" | ")
        for i, loc in enumerate(locs):
            if "[VISUAL]" in loc or "TBD" in loc:
                manual.append("%s %s" % (r["claim_id"], loc))
                continue
            snap = re.search(r"\(SNAPSHOT ([^)]+)\)", loc)
            if snap:   # web source: keywords must be in the saved raw page
                text = squash(re.sub(r"<[^>]+>", " ", html.unescape((LIT / "registry_cache" / snap.group(1)).read_text())))
                missing = [k for k in (kws[i] if i < len(kws) else "").split(";") if k and squash(k) not in text]
                n_checked += 1
                if missing:
                    fails.append("%s: snapshot %s missing keywords %s" % (r["claim_id"], snap.group(1), missing))
                continue
            m = re.match(r"(\w+):.*\(PDF p(\d+)\)\s*$", loc)
            if not m:
                continue
            key, page = m.group(1), int(m.group(2))
            f = sources.get(key, {}).get("file")
            if not f:
                fails.append("%s: %s has no local PDF" % (r["claim_id"], key))
                continue
            text = pages(f)[page - 1]
            missing = [k for k in (kws[i] if i < len(kws) else "").split(";") if k and squash(k) not in text]
            n_checked += 1
            if missing:
                fails.append("%s: %s PDF p%d missing keywords %s" % (r["claim_id"], key, page, missing))
        for key in filter(None, r["bib_key"].split(";")):
            if key not in bib:
                fails.append("%s: bib_key %s not in references.bib" % (r["claim_id"], key))

    # 3. deck citations resolve
    cites = {k.strip() for grp in re.findall(r"\\cite\w*\{([^}]*)\}", DECK.read_text()) for k in grp.split(",")}
    fails += ["deck cites unknown key %s" % k for k in sorted(cites - set(bib))]

    print("bib entries: %d | ledger locators keyword-checked: %d | deck \\cite keys: %d" % (len(bib), n_checked, len(cites)))
    print("\nNeeds manual check (%d):" % len(manual))
    for m in manual:
        print("  - " + m)
    print("\nFailures (%d):" % len(fails))
    for f in fails:
        print("  x " + f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
