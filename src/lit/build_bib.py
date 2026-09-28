"""
Build docs/lit_review/references.bib from registry records only.

Every field comes from a registry response (Crossref, DataCite, arXiv API,
or the publisher's own BibTeX), never from memory. Raw responses are cached
in docs/lit_review/registry_cache/ so each field can be traced and the bib
rebuilt offline. Entries with no registry (books without DOI, lecture notes,
a thesis, KDD-96) are in MANUAL with a note on where each field was read.

Usage: python3 src/lit/build_bib.py [--refresh]
"""
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIT = ROOT / "docs" / "lit_review"
CACHE = LIT / "registry_cache"
REFRESH = "--refresh" in sys.argv
CITABLE = {"IN_HAND", "ATTRIBUTION_ONLY", "BOOK_NO_PDF", "WEB_SNAPSHOT"}

# publisher BibTeX for venues without DOIs
PUBLISHER_BIB = {
    "carrieremicheloudot2018": "https://www.jmlr.org/papers/v19/17-291.bib",
    "pedregosa2011sklearn": "https://www.jmlr.org/papers/v12/pedregosa11a.bib",
    "levinabickel2004": "https://proceedings.neurips.cc/paper_files/paper/2004/file/74934548253bcab8490ebd74afed7031-Bibtex.bib",
}

# no registry record exists; each field was read from the named source
MANUAL = {
    "hatcher2001": ("book", dict(author="Hatcher, Allen", title="Algebraic Topology", publisher="Cambridge University Press",
                                 year="2001", isbn="0521795400", url="https://pi.math.cornell.edu/~hatcher/AT/AT.pdf",
                                 note="Page numbers refer to the author's free online edition (PDF generated 2022-10-26)",
                                 fieldsource="Open Library edition records for ISBN 0521795400 and 052179160X (CUP, 2001-11-15); "
                                             "the online PDF has no text title/copyright page")),
    "hoehn2018notes": ("misc", dict(author="H{\\\"o}hn, Gerald", title="Algebraic Topology Lecture Notes",
                                    howpublished="Lecture notes",
                                    url="https://www.math.ksu.edu/~gerald/math875c/algebraic-topology.pdf",
                                    note="Accessed 2026-09-28; title, author, and terms read from PDF title page; institution not stated in the notes")),
    "ester1996dbscan": ("inproceedings", dict(author="Ester, Martin and Kriegel, Hans-Peter and Sander, J{\\\"o}rg and Xu, Xiaowei",
                                              title="A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
                                              booktitle="Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)",
                                              pages="226--231", publisher="AAAI Press", year="1996",
                                              url="https://cdn.aaai.org/KDD/1996/KDD96-037.pdf",
                                              note="Fields from aaai.org paper page; pages from the PDF")),
    "munkres2000topology": ("book", dict(author="Munkres, James R.", title="Topology", edition="2nd", publisher="Prentice Hall",
                                         year="2000", isbn="0131816292", note="Open Library ISBN record")),
    "mackay2005comments": ("misc", dict(author="MacKay, David J. C. and Ghahramani, Zoubin",
                                        title="Comments on `Maximum Likelihood Estimation of Intrinsic Dimension' by E. Levina and P. Bickel (2004)",
                                        howpublished="Web note", url="http://www.inference.org.uk/mackay/dimension/", year="2005",
                                        note="Accessed 2026-09-28; year from the page's 'Last modified' line; snapshot in registry_cache")),
    "sklearn_userguide_manifold": ("misc", dict(author="{scikit-learn developers}", title="Manifold learning (User Guide, Sec. 2.2)",
                                                howpublished="scikit-learn 1.9.1 documentation", url="https://scikit-learn.org/stable/modules/manifold.html",
                                                year="2026", note="Accessed 2026-09-28; owner from page footer; snapshot in registry_cache")),
    "keplermapper_repo": ("misc", dict(author="{scikit-tda}", title="Kepler Mapper (GitHub repository)", url="https://github.com/scikit-tda/kepler-mapper",
                                       year="2026", note="Accessed 2026-09-28; README snapshot in registry_cache")),
    "reeb1946": ("article", dict(author="Reeb, G.",
                                 title="Sur les points singuliers d'une forme de {P}faff compl{\\`e}tement int{\\'e}grable ou d'une fonction num{\\'e}rique",
                                 journal="Comptes Rendus de l'Acad{\\'e}mie des Sciences de Paris", volume="222", pages="847--849", year="1946",
                                 note="Not read; fields from three concordant citations (Carri{\\`e}re--Oudot ref. 31; Munch--Wang ref. 21; Lum et al. ref. 6)")),
    "mohnhaupt2023": ("mastersthesis", dict(author="Mohnhaupt, Mona", title="The Nerve Theorem and its Applications in Topological Data Analysis",
                                            school="ETH Z{\\\"u}rich", year="2023", type="Bachelor's thesis",
                                            note="Read from PDF title page (supervisor: S. Kali{\\v{s}}nik Hintz)")),
}


def fetch(url, name):
    """Fetch with an on-disk cache; the cache is the audit trail."""
    path = CACHE / name
    if path.exists() and not REFRESH:
        return path.read_text()
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-bib-builder (academic use)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8")
    path.write_text(body)
    time.sleep(1)
    return body


# fields a registry leaves blank, filled from the source PDF itself (provenance noted)
OVERRIDES = {
    "edelsbrunnerletscherzomorodian2002": dict(
        author="Edelsbrunner, Herbert and Letscher, David and Zomorodian, Afra",
        fieldsource="author given names from the published PDF title page (Crossref lists surnames only)"),
    "axler2024ladr": dict(edition="4th", series="Undergraduate Texts in Mathematics",
                          fieldsource="edition from PDF footer; series from Crossref container-title"),
    "singh2007mapper": dict(author="Singh, Gurjeet and M{\\'e}moli, Facundo and Carlsson, Gunnar",
                            fieldsource="accent from the published PDF (DataCite has 'Memoli')"),
    "hoehn2018notes": dict(year="2018", fieldsource="year = latest term on the notes' title page"),
}


def tex(s):
    s = re.sub(r"<[^>]+>", "", s)                     # Crossref titles can carry HTML tags
    s = re.sub(r"\s+", " ", s).replace("&amp;", "&").strip()
    return re.sub(r"(?<!\\)&", r"\\&", s)


def from_crossref(doi):
    m = json.loads(fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi), "crossref_" + doi.replace("/", "_") + ".json"))["message"]
    kind = {"journal-article": "article", "proceedings-article": "inproceedings", "book": "book",
            "monograph": "book", "book-chapter": "incollection"}.get(m["type"], "misc")
    f = dict(author=" and ".join("%s, %s" % (a["family"], a.get("given", "")) for a in m.get("author", [])),
             title=tex(m["title"][0]), doi=m["DOI"])
    ct = (m.get("container-title") or [""])[0]
    if kind == "article":
        f["journal"] = tex(ct)
    elif kind in ("inproceedings", "incollection"):
        f["booktitle"] = tex(ct)
    for k in ("volume", "issue", "page", "publisher"):
        if m.get(k):
            f[{"issue": "number", "page": "pages"}.get(k, k)] = tex(m[k]).replace("-", "--")
    if m.get("article-number") and "pages" not in f:
        f["pages"] = m["article-number"]
    issued = (m.get("published-print") or m.get("issued"))["date-parts"][0]
    f["year"] = str(issued[0])
    return kind, f


def from_datacite(doi):
    a = json.loads(fetch("https://api.datacite.org/dois/" + urllib.parse.quote(doi), "datacite_" + doi.replace("/", "_") + ".json"))["data"]["attributes"]
    c = a.get("container") or {}
    f = dict(author=" and ".join("%s, %s" % (x.get("familyName", x.get("name")), x.get("givenName", "")) for x in a["creators"]),
             title=tex(a["titles"][0]["title"]), year=str(a["publicationYear"]), publisher=tex(a.get("publisher", "")), doi=a["doi"])
    if c.get("title"):
        f["booktitle"] = tex(c["title"])
    if c.get("volume"):
        f["volume"] = c["volume"]
    if c.get("firstPage"):
        f["pages"] = c["firstPage"] + ("--" + c["lastPage"] if c.get("lastPage") else "")
    return "inproceedings", f


def from_arxiv(aid):
    t = fetch("https://export.arxiv.org/api/query?id_list=" + aid, "arxiv_" + aid + ".xml")
    e = t.split("<entry>")[1]
    g = lambda p: re.search(p, e, re.S).group(1).strip()
    names = re.findall(r"<name>(.*?)</name>", e)
    f = dict(author=" and ".join("%s, %s" % (n.split()[-1], " ".join(n.split()[:-1])) for n in names),
             title=tex(g(r"<title>(.*?)</title>")), year=g(r"<updated>(\d{4})"),   # date of the version held
             eprint=re.sub(r"v\d+$", "", aid), archiveprefix="arXiv", note="arXiv:" + aid)
    return "misc", f


def from_publisher_bib(key):
    body = fetch(PUBLISHER_BIB[key], "publisher_" + key + ".bib")
    kind = re.match(r"\s*@(\w+)", body).group(1).lower()
    f = {k.lower(): tex(v) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S)}
    return kind, f


def from_snapshot(row):
    """Web source: title and owner read from the saved raw page."""
    snap = re.search(r"snapshot=(\S+?\.html)", row["notes"]).group(1)
    raw = (CACHE / snap).read_text()
    title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S).group(1)
    owner = "MLB Advanced Media" if "MLB Advanced Media" in raw else "unknown"
    return "misc", dict(author="{%s}" % owner, title=tex(html.unescape(title)), url=row["identifier"], year=snap[-15:-11],
                        note="Accessed %s; snapshot in docs/lit_review/registry_cache/%s" % (snap[-15:-5], snap))


def entry(key, row):
    ident = row["identifier"]
    if key in MANUAL:
        return MANUAL[key]
    if ident.startswith("pypi:"):
        name, ver = ident[5:].split("==")
        i = json.loads(fetch("https://pypi.org/pypi/%s/%s/json" % (name, ver), "pypi_%s_%s.json" % (name, ver)))
        return "software", dict(author=i["info"]["author"], title=name, version=ver, url=i["info"]["home_page"],
                                year=i["urls"][0]["upload_time"][:4], note="PyPI release %s (%s license)" % (ver, i["info"]["license"]))
    if row["status"] == "WEB_SNAPSHOT":
        return from_snapshot(row)
    if key in PUBLISHER_BIB:
        return from_publisher_bib(key)
    if ident.startswith("doi:"):
        doi = ident[4:]
        return from_datacite(doi) if doi.lower().startswith(("10.4230/", "10.2312/")) else from_crossref(doi)
    if ident.startswith("arXiv:"):
        return from_arxiv(ident[6:])
    raise SystemExit("no registry route for %s (%s)" % (key, ident))


def main():
    CACHE.mkdir(exist_ok=True)
    rows = [r for r in csv.DictReader(open(LIT / "sources.csv")) if r["bib_key"] and r["status"] in CITABLE]
    out = ["% Generated by src/lit/build_bib.py from registry records. Do not edit by hand.\n"]
    for r in rows:
        kind, f = entry(r["bib_key"], r)
        f = {**f, **OVERRIDES.get(r["bib_key"], {})}
        body = ",\n".join("  %s = {%s}" % (k, v) for k, v in f.items() if v)
        out.append("@%s{%s,\n%s\n}\n" % (kind, r["bib_key"], body))
    (LIT / "references.bib").write_text("\n".join(out))
    print("wrote %d entries to %s" % (len(rows), LIT / "references.bib"))


if __name__ == "__main__":
    main()
