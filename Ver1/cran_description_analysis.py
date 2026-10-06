#!/usr/bin/env python3
r"""Find why DESCRIPTION cannot distinguish CRAN Precon from CRAN Postcon.

Save in ewoc_ticket_type and run with the project's existing Python environment:
    .\.venv\Scripts\python.exe .\cran_description_analysis.py

Reads original cached work orders for the same 18-month WHERE clause used in
the screenshots. If a cache is missing, the existing data_loader fetches it
from Oracle. --refresh explicitly fetches again. Uses the project's CURRENT
normalize_description and insufficient_information functions, never a copied
normalizer and never cleaned_data.csv as the before-cleaning source.

Optional original export:
    python cran_description_analysis.py --raw-csv original_work_orders.csv
    python cran_description_analysis.py --raw-csv original.csv --type-csv types.csv

Produces ONE small HTML report and a short terminal summary. No model fitting,
MLflow registration, row dumps, or additional packages beyond the project.
Phrase counts use document presence, not repeated occurrences within a ticket.
CountVectorizer API: https://scikit-learn.org/stable/modules/generated/
sklearn.feature_extraction.text.CountVectorizer.html
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import inspect
import logging
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


PRE, POST = "CRAN Precon", "CRAN Postcon"
CLASSES = [PRE, POST]
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
PRE_CUE = re.compile(r"\bpre[\s_-]*con(?:struction)?\b", re.I)
POST_CUE = re.compile(r"\bpost[\s_-]*con(?:struction)?\b", re.I)
IGNORE_REMOVED = set("usid fa original ticket work order id number no numid emailid ttid".split())
GENERIC = set("a an the and or for of to in on with please review task nrc numid emailid ttid".split())


def canonical(text):
    """Ignore only case and whitespace; preserve words, numbers and punctuation."""
    return " ".join(str(text).lower().split())


def mask_ids(text, keep_lines=False):
    """Exploratory view: preserve wording but replace common identifier shapes.

    Long numbers may include meaningful values; this is NOT a new preprocessing
    recommendation or proof that every masked value is an identifier.
    """
    text = EMAIL.sub(" emailid ", str(text).lower())
    text = re.sub(r"\btt\d+\b", " ttid ", text)
    text = re.sub(r"\b\d{5,}\b", " numid ", text)
    return "\n".join(canonical(line) for line in text.splitlines()) if keep_lines else canonical(text)


def loss_words(raw, normalized):
    # Exclude email text so removal of an email name is not a 'lost content word'.
    before = Counter(re.findall(r"\b[a-z]+\b", EMAIL.sub(" ", raw.lower())))
    after = Counter(re.findall(r"\b[a-z]+\b", normalized.lower()))
    return set(before - after) - IGNORE_REMOVED


def overlap(frame, column):
    counts = pd.crosstab(frame[column], frame.label).reindex(columns=CLASSES, fill_value=0)
    counts["total"] = counts[PRE] + counts[POST]
    shared = counts[(counts[PRE] > 0) & (counts[POST] > 0)].copy()
    forced_errors = int(counts[CLASSES].min(axis=1).sum())
    return {
        "unique": len(counts), "shared_groups": len(shared),
        "shared_rows": int(shared.total.sum()), "forced_errors": forced_errors,
        "ceiling": 1 - forced_errors / len(frame),
        "counts": counts, "shared": shared.sort_values("total", ascending=False),
    }


def phrase_analysis(frame, column):
    """Frequent and distinguishing 1-5 word phrases, weighted by ticket count."""
    min_docs = max(2, math.ceil(len(frame) * .001))
    vectorizer = CountVectorizer(binary=True, lowercase=True, ngram_range=(1, 5),
                                token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=min_docs,
                                max_features=50000)
    try:
        matrix = vectorizer.fit_transform(frame[column])
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "no terms remain" in str(exc).lower():
            return [], [], min_docs
        raise
    terms = vectorizer.get_feature_names_out()
    npre, npost = int((frame.label == PRE).sum()), int((frame.label == POST).sum())
    cpre = np.asarray(matrix[(frame.label == PRE).to_numpy()].sum(axis=0)).ravel()
    cpost = np.asarray(matrix[(frame.label == POST).to_numpy()].sum(axis=0)).ravel()
    candidates = []
    for term, a, b in zip(terms, cpre, cpost):
        words = term.split()
        candidates.append({"text": term, "pre": int(a), "post": int(b),
                           "rp": float(a / npre), "rq": float(b / npost),
                           "gap": float(a / npre - b / npost),
                           "words": len(words), "content": bool(set(words) - GENERIC)})

    def select(pool, limit, score):
        chosen = []
        for item in sorted(pool, key=score, reverse=True):
            # Suppress nested n-grams with essentially the same coverage.
            if any((f" {item['text']} " in f" {old['text']} " or
                    f" {old['text']} " in f" {item['text']} ") and
                   abs(item["rp"] - old["rp"]) <= .03 and
                   abs(item["rq"] - old["rq"]) <= .03 for old in chosen):
                continue
            chosen.append(item)
            if len(chosen) == limit:
                break
        return chosen

    frequent = select([x for x in candidates if x["words"] >= 2], 4,
                      lambda x: (x["pre"] + x["post"], x["words"]))
    distinct = []
    for sign in [1, -1]:
        distinct.extend(select([x for x in candidates if x["content"] and sign*x["gap"] >= .05],
                               3, lambda x: (abs(x["gap"]), x["words"], x["pre"] + x["post"])))
    return frequent, distinct, min_docs


def fragments(frame, column):
    """Simple sentence/clause fragments, counted at most once per ticket."""
    totals = {label: Counter() for label in CLASSES}
    source_column = "raw" if column == "masked" else column
    for (text, label), weight in frame.groupby([source_column, "label"]).size().items():
        if column == "masked":
            text = mask_ids(text, keep_lines=True)
        parts = {canonical(s).strip() for s in re.split(r"[.!?;,\n]+", text)}
        parts = {s for s in parts if len(re.findall(r"[a-z]+", s)) >= 3}
        totals[label].update({s: int(weight) for s in parts})
    all_parts = set(totals[PRE]) | set(totals[POST])
    ordered = sorted(all_parts, key=lambda s: (-(totals[PRE][s]+totals[POST][s]), s))
    return [(s, totals[PRE][s], totals[POST][s]) for s in ordered[:4]]


def prepare_frame(raw, label_column, text_column, date_column, loader):
    names = raw[label_column].fillna("").map(canonical)
    selected = raw[names.isin([canonical(x) for x in CLASSES])].copy()
    labels = names.loc[selected.index].map({canonical(x): x for x in CLASSES})
    frame = pd.DataFrame({"label": labels, "raw": selected[text_column].fillna("").astype(str)})
    frame["source_row"] = np.arange(len(raw))[names.isin([canonical(x) for x in CLASSES])] + 2
    blanks = frame.raw.str.strip().eq("")
    missing = frame[blanks].label.value_counts().to_dict()
    frame = frame[~blanks].copy()
    if frame.label.nunique() != 2:
        raise ValueError("Need nonempty original descriptions for BOTH CRAN Precon and CRAN Postcon")
    if date_column in selected:
        frame["date"] = pd.to_datetime(selected.loc[frame.index, date_column], errors="coerce")
    frame = frame.reset_index(drop=True)
    frame["raw_key"] = frame.raw.map(canonical)
    frame["masked"] = frame.raw.map(mask_ids)
    frame["normalized"] = frame.raw.map(loader.normalize_description)
    if not frame.normalized.map(lambda x: isinstance(x, str)).all():
        raise TypeError("normalize_description must return strings")
    frame["rejected"] = frame.raw.map(loader.insufficient_information).astype(bool)
    frame["lost_words"] = [loss_words(a, b) for a, b in zip(frame.raw, frame.normalized)]
    frame["pre_raw"] = frame.raw.str.contains(PRE_CUE)
    frame["post_raw"] = frame.raw.str.contains(POST_CUE)
    frame["pre_norm"] = frame.normalized.str.contains(PRE_CUE)
    frame["post_norm"] = frame.normalized.str.contains(POST_CUE)
    frame["stage_lost"] = ((frame.pre_raw & ~frame.pre_norm) | (frame.post_raw & ~frame.post_norm))
    return frame, missing


def diagnose(frame, metrics):
    n = len(frame)
    original, masked, norm = (metrics[x] for x in ["raw_key", "masked", "normalized"])
    findings = []
    if norm["shared_rows"]:
        top_text, top = next(iter(norm["shared"].iterrows()))
        findings.append(f"Largest shared normalized description: {int(top.total):,} tickets "
                        f"({int(top[PRE]):,} Precon; {int(top[POST]):,} Postcon): {short(top_text, 140)}")
    if frame.stage_lost.mean() >= .05 and norm["forced_errors"] > original["forced_errors"]:
        findings.append("Priority issue to inspect: normalization removes explicit construction-stage "
                        "wording from at least 5% of the rows while increasing exact-text label conflicts. "
                        "Review the paired examples before accepting the current cleanup rules.")
    elif original["shared_rows"] / n >= .2:
        findings.append("Main observed issue: substantial label ambiguity already exists in raw wording "
                        "(ignoring case/whitespace). Normalization is not its sole cause.")
    elif norm["shared_rows"] / n >= .2 and masked["shared_rows"] / n >= .2:
        findings.append("Main observed issue: shared templates after masking identifier-shaped values. "
                        "Many raw differences are numbers/IDs rather than different wording; inspect "
                        "the examples before deciding whether those values contain usable meaning.")
    elif norm["forced_errors"] > original["forced_errors"]:
        findings.append("Normalization introduces additional exact-text label conflicts. Inspect the "
                        "removed words and paired examples to determine whether useful content is lost.")
    else:
        findings.append("Exact repeated-text conflicts do not establish a dominant cause in this snapshot. "
                        "Use the distinguishing phrases below to investigate wording and label consistency.")
    findings.append(f"Identical normalized inputs force at least {norm['forced_errors']:,}/{n:,} errors "
                    f"({norm['forced_errors']/n:.1%}) for a deterministic, single-label text-only model on these "
                    "observed rows. This is a sample-specific ambiguity bound, not test accuracy.")
    stage_lost = int(frame.stage_lost.sum())
    wording_lost = int(frame.lost_words.map(bool).sum())
    findings.append(f"Cleanup removes explicit precon/postcon cues in {stage_lost:,} tickets; other "
                    f"alphabetic words beyond common ID/email fields are removed in {wording_lost:,}. "
                    "A removed word is a review candidate, not automatically useful information.")
    findings.append(f"Current insufficient-information check rejects {int(frame.rejected.sum()):,} "
                    "of these nonempty CRAN descriptions. Rejected rows remain in this diagnostic.")
    return findings


def short(value, limit=220):
    value = " ".join(str(value).split())
    if not value:
        return "[empty]"
    return value if len(value) <= limit else value[:limit-1] + "â¦"


def table(headers, rows):
    def cell(value):
        return html.escape(str(value))
    rows = list(rows)
    if not rows:
        return "<p class='muted'>No qualifying items.</p>"
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{cell(x)}</th>" for x in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(x)}</td>" for x in row) +
                                        "</tr>" for row in rows) + "</tbody></table></div>"


def build_report(frame, missing, source, normalizer_info):
    metrics = {c: overlap(frame, c) for c in ["raw", "raw_key", "masked", "normalized"]}
    findings = diagnose(frame, metrics)
    n = len(frame)
    blocks = ["<h1>CRAN description diagnosis</h1>",
              "<p class='muted'>CRAN Precon vs CRAN Postcon Â· descriptive audit of the full selected snapshot</p>",
              "<section class='findings'><h2>Main findings</h2><ol>" +
              "".join(f"<li>{html.escape(s)}</li>" for s in findings) + "</ol></section>"]
    blocks += ["<h2>1. Data used</h2>", table(
        ["Class", "Nonempty descriptions", "Blank descriptions excluded", "Unique raw wording", "Unique normalized"],
        [(label, f"{len(g):,}", missing.get(label, 0), g.raw_key.nunique(), g.normalized.nunique())
         for label, g in frame.groupby("label")])]
    dates = ""
    if "date" in frame and frame.date.notna().any():
        dates = f" Date range: {frame.date.min()} to {frame.date.max()}; invalid/missing dates: {frame.date.isna().sum()}."
    blocks.append(f"<p class='muted'>{html.escape(source + dates)}</p>")

    blocks += ["<h2>2. Does normalization create ambiguity?</h2>", table(
        ["Text representation", "Distinct texts", "Texts used by both classes", "Tickets in those texts", "Forced errors"],
        [(name, f"{metrics[col]['unique']:,}", f"{metrics[col]['shared_groups']:,}",
          f"{metrics[col]['shared_rows']:,} ({metrics[col]['shared_rows']/n:.1%})",
          f"{metrics[col]['forced_errors']:,} ({metrics[col]['forced_errors']/n:.1%})")
         for col, name in [("raw", "Original, verbatim"), ("raw_key", "Original: case/whitespace ignored"),
                           ("masked", "Original: ID-shaped values masked"), ("normalized", "Current project normalization")]]),
        "<p class='muted'>Forced errors = sum of the smaller class count for each identical text. "
        "Unique IDs can make raw text appear separable without providing a reusable language signal. "
        "ID masking replaces emails, TT+digits and standalone numbers of 5+ digits; some such numbers "
        "may be meaningful. This exploratory view does not change your normalizer.</p>"]
    top_shared = metrics["normalized"]["shared"].head(3)
    blocks += ["<h3>Largest conflicting normalized descriptions</h3>", table(
        ["Normalized description", "Precon", "Postcon", "Raw wording variants", "Variants after ID masking"],
        [(short(text), int(row[PRE]), int(row[POST]),
          frame.loc[frame.normalized == text, "raw_key"].nunique(),
          frame.loc[frame.normalized == text, "masked"].nunique()) for text, row in top_shared.iterrows()])]

    if frame.rejected.any() and (~frame.rejected).any():
        eligible = overlap(frame[~frame.rejected], "normalized")
        blocks.append(f"<p>Among {int((~frame.rejected).sum()):,} rows that pass the current input check, "
                      f"{eligible['shared_rows']:,} still belong to shared normalized texts, forcing "
                      f"at least {eligible['forced_errors']:,} errors on those observed rows.</p>")

    blocks.append("<h2>3. Repeated phrases and sentences</h2><p>Counts are tickets containing the item, "
                  "at most once per ticket. Subphrases with nearly identical coverage are suppressed.</p>")
    for col, name in [("masked", "Before normalization (ID-shaped values masked for comparison)"),
                      ("normalized", "After current normalization")]:
        frequent, distinct, min_docs = phrase_analysis(frame, col)
        blocks.append(f"<details{' open' if col == 'normalized' else ''}><summary>{name}</summary>")
        blocks += ["<h3>Most repeated word phrases</h3>", table(
            ["Phrase", "Precon tickets", "Postcon tickets"],
            [(x["text"], f"{x['pre']:,} ({x['rp']:.1%})", f"{x['post']:,} ({x['rq']:.1%})") for x in frequent]),
            "<h3>Most repeated sentence/clause fragments</h3>", table(
                ["Fragment", "Precon tickets", "Postcon tickets"],
                [(short(t), a, b) for t, a, b in fragments(frame, col)]),
            "<h3>Phrases that distinguish the classes</h3>", table(
                ["Phrase", "More common in", "% of Precon", "% of Postcon", "Difference"],
                [(x["text"], "Precon" if x["gap"] > 0 else "Postcon", f"{x['rp']:.1%}",
                  f"{x['rq']:.1%}", f"{abs(x['gap'])*100:.1f} percentage points") for x in distinct]),
            f"<p class='muted'>Phrases: 1â5 words, at least {min_docs} tickets, maximum 50,000 candidates. "
            "Distinguishing phrases need a 5-percentage-point prevalence gap; absence from this short list "
            "does not prove no weaker signal exists. Sentence/clause splitting uses punctuation, not a language model. "
            "Percentages use each class's own ticket count.</p></details>"]

    cue_rows = []
    for label, g in frame.groupby("label"):
        for stage, a, b in [("Before", "pre_raw", "post_raw"), ("After", "pre_norm", "post_norm")]:
            cue_rows.append((label, stage, int((g[a] & ~g[b]).sum()), int((g[b] & ~g[a]).sum()),
                             int((g[a] & g[b]).sum()), int((~g[a] & ~g[b]).sum())))
    removed = Counter()
    for words in frame.lost_words:
        removed.update(words)
    blocks += ["<h2>4. What information is removed?</h2>", table(
        ["Class", "Stage", "Precon cue only", "Postcon cue only", "Both cues", "Neither cue"], cue_rows),
        "<p class='muted'>Cues include precon, pre-con, pre construction, postcon and equivalent spellings. "
        "A cue need not agree with the label; this table does not automatically relabel tickets.</p>",
        table(["Most common removed alphabetic word", "Tickets losing it"], removed.most_common(5))]

    example_ids = []
    if len(top_shared):
        largest = top_shared.index[0]
        for label in CLASSES:
            group = frame[(frame.normalized == largest) & (frame.label == label)]
            variants = group.drop_duplicates("raw_key")
            example_ids.extend(variants.head(1).index.tolist())
            if len(variants) > 1:
                example_ids.extend(variants.tail(1).index.tolist())
    extras = frame[frame.stage_lost | frame.lost_words.map(bool)].drop_duplicates(["label", "raw_key"])
    example_ids += [i for i in extras.index if i not in example_ids][:2]
    if not example_ids:
        for label in CLASSES:
            example_ids += frame[frame.label == label].head(1).index.tolist()
    examples = frame.loc[example_ids[:6]]
    blocks += ["<details><summary>5. A few paired examples (maximum six)</summary>", table(
        ["Class / source row", "Original description", "After normalization", "Removed words to inspect"],
        [(f"{r.label} / {r.source_row}", short(r.raw, 450), short(r.normalized, 450),
          ", ".join(sorted(r.lost_words)) or "â") for r in examples.itertuples()]),
        "<p class='muted'>Display text may be shortened; all calculations use complete descriptions. "
        "Source row is the data-record ordinal plus the header, not a physical CSV line for multiline records.</p></details>",
        "<h2>What to do with the result</h2><ul>"
        "<li>If conflicts already exist in original wording, check the labeling definition and whether the "
        "description contains enough context for a human to distinguish the classes.</li>"
        "<li>If cleanup removes distinguishing words or meaningful technical values, adjust those specific "
        "normalization rules before retraining.</li>"
        "<li>If differences are only opaque IDs, keeping IDs may enable memorization; it does not establish "
        "generalization to new task IDs. An ambiguous description needs more information.</li>"
        "<li>Phrase associations describe this snapshot. Test any model change on separate validation and "
        "final test data; do not treat these full-snapshot statistics as model accuracy.</li></ul>",
        f"<p class='muted'>Normalizer: {html.escape(normalizer_info)}. No training class-count cutoff was applied. "
        "Blank original descriptions are excluded; current-rule rejections and invalid dates remain visible. "
        "Conflicting labels can reflect legitimate context differences, labeling problems, or lost information.</p>"]
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>CRAN description diagnosis</title>
<style>body{font:15px/1.5 system-ui,Arial,sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#192536;background:#fff}
h1{font-size:27px;margin-bottom:4px}h2{font-size:20px;margin-top:28px}h3{font-size:16px}.muted{color:#526172;font-size:13px}
.findings{background:#eef4fa;border-left:4px solid #295e8a;padding:8px 20px}li{margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #d9e1e9;vertical-align:top;overflow-wrap:anywhere}
th{background:#f3f6f9}details{margin:16px 0;border:1px solid #d9e1e9;border-radius:6px;padding:12px}summary{font-weight:650;cursor:pointer}
.scroll{overflow-x:auto}@media print{body{max-width:none;font-size:12px}.findings{break-inside:avoid}}</style><body>"""
    return page + "\n".join(blocks) + "</body></html>", findings


def cache_or_fetch(loader, table_name, where, refresh):
    suffix = "_" + hashlib.sha256(where.encode()).hexdigest()[:12] if where else ""
    path = Path(loader.config.DATA_CACHE_DIR) / (table_name.replace(".", "_") + suffix + ".csv")
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=str, keep_default_na=False), f"Original cache: {path.resolve()}"
    result = loader.fetch_full_table(table_name, where_clause=where, force_refresh=refresh)
    return result, f"Fetched original table: {table_name}; filter: {where or 'none'}"


def key_id(series):
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--raw-csv", type=Path, help="Original unnormalized work orders; never cleaned_data.csv")
    parser.add_argument("--type-csv", type=Path, help="Optional lookup with TYPE_ID and TYPE_WO")
    parser.add_argument("--description-col", default="DESCRIPTION")
    parser.add_argument("--label-col", default="WorkOrder_Info")
    parser.add_argument("--date-col", default="CREATED_DATE")
    parser.add_argument("--work-orders-table", default="e911.ENMT_E911_WORK_ORDERS")
    parser.add_argument("--type-table", default="e911.LU_EWOC_TYPE")
    parser.add_argument("--where", default="WHERE CREATED_DATE >= ADD_MONTHS(SYSDATE, -18)")
    parser.add_argument("--refresh", action="store_true", help="Fetch source tables again through the existing loader")
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw_path = args.raw_csv.resolve() if args.raw_csv else None
    type_path = args.type_csv.resolve() if args.type_csv else None
    if raw_path and raw_path.name.lower() == "cleaned_data.csv":
        parser.error("cleaned_data.csv has already lost the original descriptions. Use the original ExtractedData cache.")
    if not (root / "src/ewoc_ttype/ewoc_utils/data_loader.py").is_file():
        parser.error("Save this script in ewoc_ticket_type, or pass --project-root pointing to that project")
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from ewoc_ttype.ewoc_utils import data_loader
    logging.getLogger("oracledb").setLevel(logging.WARNING)
    if raw_path:
        raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
        source = f"Original CSV: {raw_path}; all rows in this supplied export (no --where re-filtering)."
    else:
        raw, source = cache_or_fetch(data_loader, args.work_orders_table, args.where, args.refresh)
    raw = raw.reset_index(drop=True)
    if args.description_col not in raw:
        raise ValueError(f"Missing description column {args.description_col}")
    if args.label_col not in raw:
        if "TYPE_ID" not in raw:
            raise ValueError(f"Need {args.label_col} or TYPE_ID in the original data")
        if type_path:
            types = pd.read_csv(type_path, dtype=str, keep_default_na=False)
        else:
            types, _ = cache_or_fetch(data_loader, args.type_table, None, args.refresh)
        if not {"TYPE_ID", "TYPE_WO"}.issubset(types.columns):
            raise ValueError("Type lookup must contain TYPE_ID and TYPE_WO")
        keys = key_id(types.TYPE_ID)
        mapping = pd.DataFrame({"key": keys, "name": types.TYPE_WO}).drop_duplicates()
        if mapping.key.duplicated().any():
            raise ValueError("Conflicting TYPE_WO names for the same TYPE_ID; resolve the lookup first")
        raw[args.label_col] = key_id(raw.TYPE_ID).map(mapping.set_index("key").name)
    print("Analyzing original CRAN descriptions with the project's current normalization...")
    frame, missing = prepare_frame(raw, args.label_col, args.description_col, args.date_col, data_loader)
    try:
        function_source = inspect.getsource(data_loader.normalize_description)
        digest = hashlib.sha256(function_source.encode()).hexdigest()[:12]
    except (OSError, TypeError):
        digest = "unavailable"
    normalizer_info = f"{Path(data_loader.__file__).resolve()} Â· normalize_description SHA256 {digest}"
    report, findings = build_report(frame, missing, source, normalizer_info)
    out = root / "EWOCTypePredArtifacts" / "cran_analysis"
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = out / f"cran_description_report_{timestamp}.html"
    path.write_text(report, encoding="utf-8")
    print(f"Analyzed {len(frame):,} nonempty descriptions from both CRAN classes.")
    for finding in findings:
        print("- " + finding)
    print(f"\nONE report saved to: {path}")
    print("Open this HTML file in a browser. The Before section and paired examples expand on click.")


if __name__ == "__main__":
    main()
