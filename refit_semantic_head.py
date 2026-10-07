"""Reuse the latest saved validation checkpoint; never fine-tune its encoder.

Save beside the updated main_train.py. From the EWOC project directory run:
    python src/ewoc_ttype/refit_semantic_head.py
An optional positional argument selects a specific human-ticket run folder.
Stop the original training process first so its checkpoint cannot change mid-read.
"""
import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    import main_train as training
else:
    from . import main_train as training


@torch.inference_mode()
def classify(model, features):
    model.eval()
    return model.head(torch.from_numpy(features)).argmax(1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    training.logger.setLevel(logging.INFO)
    training.disable_progress_bar()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    runs = sorted(path for path in (training.ROOT / "EWOCTypePredArtifacts").glob("human_tickets_*")
                  if (path / "transformer/validation_checkpoint/classifier_config.json").exists())
    if args.run is None and not runs:
        raise SystemExit("No saved validation checkpoint found. Supply the existing human-ticket run folder.")
    source = args.run if args.run is not None else runs[-1]
    checkpoint = source / "transformer/validation_checkpoint"
    frame = pd.read_csv(source / "dataset.csv", keep_default_na=False)
    train, valid, test = [frame[frame.split.eq(name)].copy() for name in ("train", "validation", "test")]
    if train.empty or valid.empty or test.empty or set(train.DESCRIPTION) & set(valid.DESCRIPTION):
        raise ValueError("Expected nonempty, disjoint training and validation text groups in the saved run.")
    labels = joblib.load(source / "label_encoder.pkl")
    y, y_valid = labels.transform(train.label), labels.transform(valid.label)
    output = source / ("head_refit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    print(f"Using checkpoint: {checkpoint.resolve()}", flush=True)
    print("Fitting classifiers only; the encoder weights stay fixed.", flush=True)
    comparison, candidates = [], []

    def record(name, model, x, xv):
        scores = training.metric_values(valid.label, labels.inverse_transform(classify(model, xv)))
        train_scores = training.metric_values(train.label, labels.inverse_transform(classify(model, x)))
        comparison.append({"model": name, "split": "validation", **scores,
                           **{"train_" + key: value for key, value in train_scores.items()}})
        path = output / "validation_checkpoint"
        if not candidates or training.selection_score(scores) > max(item["score"] for item in candidates):
            model.save(path)
        candidates.append({"name": name, "path": path, "score": training.selection_score(scores),
                           "features": x, "validation_features": xv, "settings": model.head_settings})

    for encoder_name, path in (("checkpoint", checkpoint), ("pretrained", training.ENCODER_PATH)):
        print(f"Encoding with {encoder_name} MiniLM...", flush=True)
        model = (training.TicketClassifier.load(path) if encoder_name == "checkpoint"
                 else training.TicketClassifier(path, len(labels.classes_)))
        model.eval()
        model.requires_grad_(False)
        x = training.encode_training(model, train.DESCRIPTION, f"{encoder_name}: training")
        xv = training.encode_training(model, valid.DESCRIPTION, f"{encoder_name}: validation")
        if encoder_name == "checkpoint":
            record("checkpoint_existing_head", model, x, xv)
        print("Selecting classifier regularization and weighting on validation...", flush=True)
        training.fit_semantic_head(model, x, y, validation=(xv, y_valid))
        record(encoder_name + "_tuned_head", model, x, xv)
        del model

    chosen = max(candidates, key=lambda item: item["score"])
    model = training.TicketClassifier.load(chosen["path"])
    improved = chosen["name"] != "checkpoint_existing_head"
    selection = {"source_run": str(source.resolve()), "winner": chosen["name"],
                 "head_settings": chosen["settings"], "encoder_fine_tuned_here": False,
                 "head_refit_on_train_plus_validation": improved,
                 "criterion": "minimum of validation accuracy/macro-F1/weighted-F1",
                 "classes_absent_from_validation": sorted(set(train.label) - set(valid.label))}
    (output / "selection_before_test.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    if improved:
        print(f"Refitting selected head ({chosen['name']}) on training + validation...", flush=True)
        training.fit_semantic_head(model, np.vstack([chosen["features"], chosen["validation_features"]]),
                                    np.r_[y, y_valid], settings=chosen["settings"])
    else:
        print("Neither tuned candidate improved validation; retaining the existing checkpoint.", flush=True)
    model.save(output / "text_classifier")
    ids = training.predict_texts(model, test, "Evaluating selected model")
    scores, predicted = training.evaluate(ids, test, labels, set(pd.concat([train, valid]).DESCRIPTION)
                                         if improved else set(train.DESCRIPTION))
    comparison.append({"model": "selected_semantic_model", "split": "test", **scores})
    training.save_evaluation(output, "test", test, predicted)
    for name in ("label_encoder.pkl", "type_name_to_id.json"):
        shutil.copy2(source / name, output / name)
    table = pd.DataFrame(comparison)
    table.to_csv(output / "comparison.csv", index=False)
    display = table[["model", "split", *training.METRICS]].copy()
    display[training.METRICS] = display[training.METRICS].map(lambda value: f"{value:.1%}")
    print("\n" + display.to_string(index=False))
    print(f"Validation winner: {chosen['name']}; head settings: {chosen['settings']}")
    print(f"Unseen test labels: {scores['unseen_label_rows']}/{len(test)}")
    print(f"Files: {output.resolve()}")


if __name__ == "__main__":
    main()
