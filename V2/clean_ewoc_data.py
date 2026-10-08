#!/usr/bin/env python3
"""Create an auditable, versioned EWOC CSV. The source file is never overwritten.

Run from the project root: python src/ewoc_ttype/clean_ewoc_data.py
Default input: data/whole_ewoc_tickets.csv. Only pandas is required.
Rules are business rules supplied by the EWOC owner, not model predictions.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RULESET = "ewoc_cleanup_v1"
NRC_TYPES = {32, 33, 34, 35}
TYPE_NAMES = {2: "E911 Provisioning", 5: "Other", 6: "Prelaunch Correction"}
NRC = re.compile(r"^please\s+review\s+nrc\b", re.I)
PRELAUNCH = re.compile(r"\bpre[\s-]*launch\b", re.I)
# Conservative automatic correction: action + relevant technical target in one clause.
ACTION = re.compile(
    r"\b(?:updat(?:e[ds]?|ing)|correct(?:s|ed|ing|ions?)?|fix(?:es|ed|ing)?|"
    r"(?:re)?load(?:s|ed|ing)?|missing|mismatch(?:es|ed)?|incorrect|wrong|failed|not\s+match(?:ing)?|"
    r"provision(?:s|ed|ing)?|add(?:s|ed|ing)?|inconsisten(?:t|cy))\b")
TARGET = re.compile(
    r"\b(?:databases?|db|gmlc|e[\s-]?smlc|clsg|ecgi|cgi|"
    r"cell[\s_-]*radius|cell[\s_-]*ids?|"
    r"(?:e[\s-]?911|911)\s+(?:data|records?|sectors?|provisioning)|"
    r"(?:cell|sector)\s+(?:data|records?|parameters?|entries)|data\s+fill)\b")
SPATIAL = re.compile(
    r"\b(?:lat|lon|lng|latitude|longitude|coordinates?|coordinations?|"
    r"lat[\s/-]+long|street\s+address(?:es)?|physical\s+address(?:es)?|address(?:es)?)\b")
NON_PROVISIONING = re.compile(r"\b(?:rf\s+(?:safety|exposure|shutdown)|mpe|eme|rfsrp|rfed|exposure\s+maps?)\b")
NEGATED = re.compile(
    r"\b(?:no|not|never|without|do\s+not|don't|does\s+not|doesn't|"
    r"nothing|already|previously|completed|resolved)\b")


def matching_text(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text).strip()


def provisioning_evidence(text):
    """Return an actionable clause; uncertainty stays with the original label."""
    # Keep whole-description spatial exclusions: a mixed request needs review.
    if SPATIAL.search(text):
        return "", "location_or_address_context"
    if NON_PROVISIONING.search(text):
        return "", "rf_context_needs_review"
    for clause in re.split(r"[;.!?\n]+", text):
        actions, targets = list(ACTION.finditer(clause)), list(TARGET.finditer(clause))
        for action in actions:
            for target in targets:
                start, end = min(action.start(), target.start()), max(action.end(), target.end())
                if end - start > 180:
                    continue
                context = clause[max(0, start - 60):min(len(clause), end + 60)]
                # 'not loaded' is a fault; 'do not load' is an instruction not to act.
                context = re.sub(r"\b(?:does\s+)?not\s+(?:loaded|loading|provisioned|updated|present|match(?:ing)?)\b", "missing", context)
                if not NEGATED.search(context):
                    return clause.strip()[:240], ""
    return "", "ambiguous_provisioning_candidate" if ACTION.search(text) else ""


def clean_frame(frame):
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.upper()
    required = {"TYPE_ID", "CREATED_ID", "DESCRIPTION"}
    if not required <= set(frame):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(frame))}")
    if frame.columns.duplicated().any() or "ORIGINAL_TYPE_ID" in frame:
        raise ValueError("Use the original CSV, with unique column names, not a previously cleaned export.")
    types = pd.to_numeric(frame.TYPE_ID, errors="raise")
    if types.isna().any() or types.mod(1).ne(0).any():
        raise ValueError("Every TYPE_ID must be an integer.")
    creators = pd.to_numeric(frame.CREATED_ID.replace("", pd.NA), errors="raise")
    text = frame.DESCRIPTION.fillna("").map(matching_text)
    frame["ORIGINAL_TYPE_ID"] = types.astype("int64")
    frame["TYPE_ID"] = frame.ORIGINAL_TYPE_ID.copy()
    frame["SOURCE_RECORD"] = range(1, len(frame) + 1)
    frame["CLEANUP_RULE"] = "unchanged"
    frame["CLEANUP_EVIDENCE"] = ""
    frame["CLEANUP_REVIEW_REASON"] = ""
    remove = creators.eq(1) & types.isin(NRC_TYPES) & text.str.match(NRC)
    frame.loc[remove, "CLEANUP_RULE"] = "exclude_superuser_nrc_template"

    # Types 2 and 6: literal prelaunch mention is the agreed business boundary.
    prelaunch = text.str.contains(PRELAUNCH)
    for mask, new_type, rule in (
        (~remove & types.eq(2) & prelaunch, 6, "provisioning_to_prelaunch"),
        (~remove & types.eq(6) & ~prelaunch, 2, "prelaunch_without_keyword_to_provisioning"),
    ):
        frame.loc[mask, "TYPE_ID"] = new_type
        frame.loc[mask, "CLEANUP_RULE"] = rule

    for index in frame.index[~remove & types.eq(5) & text.ne("")]:
        evidence, review = provisioning_evidence(text.at[index])
        if evidence:
            frame.at[index, "TYPE_ID"] = 6 if prelaunch.at[index] else 2
            frame.at[index, "CLEANUP_RULE"] = (
                "other_to_prelaunch" if prelaunch.at[index] else "other_to_provisioning")
            frame.at[index, "CLEANUP_EVIDENCE"] = evidence
        elif review:
            frame.at[index, "CLEANUP_REVIEW_REASON"] = review
    frame.loc[~remove & text.eq(""), "CLEANUP_REVIEW_REASON"] = "empty_description"
    # A template-like row outside the exact exclusion remains available for learning.
    retained_template = ~remove & creators.eq(1) & types.isin(NRC_TYPES)
    frame.loc[retained_template, "CLEANUP_REVIEW_REASON"] = "non_nrc_description_retained"
    # Synchronize optional existing names for corrected rows; never change other type names.
    changed = ~remove & frame.TYPE_ID.ne(frame.ORIGINAL_TYPE_ID)
    for column in ("LABEL", "TYPE_NAME", "WORK_ORDER_TYPE"):
        if column in frame:
            frame.loc[changed, column] = frame.loc[changed, "TYPE_ID"].map(TYPE_NAMES)
    audit = frame[remove | changed].copy()
    audit["ACTION"] = ["excluded" if value else "relabelled" for value in remove.loc[audit.index]]
    clean = frame[~remove].copy()
    review = clean[clean.CLEANUP_REVIEW_REASON.ne("")].copy()
    summary = {
        "ruleset": RULESET, "input_rows": len(frame), "cleaned_rows": len(clean),
        "excluded_rows": int(remove.sum()), "relabelled_rows": int(changed.sum()),
        "review_rows_retained": len(review),
        "superuser_rows_retained": int((creators.eq(1) & ~remove).sum()),
        "rule_counts": {str(k): int(v) for k, v in audit.CLEANUP_RULE.value_counts().items()},
        "description_changed": False,
        "scope": "Only creator=1 AND type in 32,33,34,35 AND Please review NRC prefix are removed.",
        "review_policy": "Review rows retain their original type unless an explicit 2/6 rule applies.",
        "label_policy": "Keyword/business-rule targets; scores against these targets do not independently validate the rules.",
    }
    return clean, audit, review, summary


def clean_csv(source, output_root):
    source = source.resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    clean, audit, review, summary = clean_frame(frame)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = output_root / f"{RULESET}_{stamp}_{source_hash[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    cleaned_file = destination / "whole_ewoc_tickets_cleaned.csv"
    clean.to_csv(cleaned_file, index=False, encoding="utf-8-sig")
    audit.to_csv(destination / "changes.csv", index=False, encoding="utf-8-sig")
    review.to_csv(destination / "review.csv", index=False, encoding="utf-8-sig")
    summary.update(source=str(source), source_sha256=source_hash, created_utc=stamp,
                   cleaned_file=str(cleaned_file.resolve()),
                   cleaned_sha256=hashlib.sha256(cleaned_file.read_bytes()).hexdigest(),
                   cleaner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pointer = output_root / "latest.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps({"cleaned_file": str(cleaned_file.resolve()),
                                     "summary": str((destination / "summary.json").resolve())}, indent=2), encoding="utf-8")
    temporary.replace(pointer)
    return cleaned_file, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/whole_ewoc_tickets.csv"))
    args = parser.parse_args()
    print("Cleaning EWOC data...", flush=True)
    path, summary = clean_csv(args.input, args.input.parent / "cleaned")
    print(f"Rows: {summary['input_rows']:,} -> {summary['cleaned_rows']:,} | "
          f"excluded: {summary['excluded_rows']:,} | relabelled: {summary['relabelled_rows']:,}")
    print(f"Superuser rows retained: {summary['superuser_rows_retained']:,} | "
          f"review candidates retained: {summary['review_rows_retained']:,}")
    print(f"Cleaned CSV: {path.resolve()}\nAudit: {path.parent.resolve() / 'changes.csv'}", flush=True)


if __name__ == "__main__":
    main()
