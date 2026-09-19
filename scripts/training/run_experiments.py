#!/usr/bin/env python3
"""
Orchestrator script to run the 6-experiment matrix (CNN & Wav2Vec2 on sentence, improvisation, and both)
and generate presentation-ready summary tables in terminal and Markdown formats.
"""

import os
import sys
import glob
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset", "ThaiSER_cleaned")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "training")

SUBSET_DISPLAY_NAMES = {
    "sentence": "Sentence (Script)",
    "script": "Sentence (Script)",
    "improvisation": "Improvisation (Imp)",
    "imp": "Improvisation (Imp)",
    "both": "Both (Script + Imp)"
}

SUBSET_NORM = {
    "sentence": "sentence",
    "script": "sentence",
    "improvisation": "improvisation",
    "imp": "improvisation",
    "both": "both"
}

MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "wav2vec": "Wav2Vec2",
    "wav2vec2": "Wav2Vec2"
}


def normalize_subset(subset):
    return SUBSET_NORM.get(subset.lower(), subset.lower())


def parse_args():
    parser = argparse.ArgumentParser(description="Run ThaiSER emotion recognition experiment matrix")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["cnn", "wav2vec", "wav2vec2"],
        default=["cnn", "wav2vec"],
        help="Models to evaluate: cnn, wav2vec (default: ['cnn', 'wav2vec'])"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=["sentence", "improvisation", "both", "script", "imp"],
        default=["sentence", "improvisation", "both"],
        help="Subsets to evaluate (default: ['sentence', 'improvisation', 'both'])"
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=f"Dataset root directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Results output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count for all runs (default: script defaults)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for all runs (default: script defaults)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate for all runs (default: script defaults)"
    )
    parser.add_argument(
        "--wav2vec_model_name",
        default="airesearch/wav2vec2-large-xlsr-53-th",
        help="Model name for Wav2Vec2 (default: airesearch/wav2vec2-large-xlsr-53-th)"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="Save best model weights (default: True)"
    )
    parser.add_argument(
        "--no_save_model",
        dest="save_model",
        action="store_false",
        help="Do not save model weights"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run only 1 train & 1 val batch per configuration to verify pipeline"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Skip training and only generate the summary table from existing result JSONs"
    )
    return parser.parse_args()


def run_experiment(model_type, subset, args):
    norm_sub = normalize_subset(subset)
    model_key = "wav2vec" if model_type in ["wav2vec", "wav2vec2"] else "cnn"
    script_name = "train_wav2vec.py" if model_key == "wav2vec" else "train_cnn.py"
    script_path = os.path.join(SCRIPTS_DIR, script_name)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [
        sys.executable,
        script_path,
        "--subset", norm_sub,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ]

    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if model_key == "wav2vec" and args.wav2vec_model_name:
        cmd.extend(["--model_name", args.wav2vec_model_name])
    if not args.save_model:
        cmd.append("--no_save_model")
    if args.dry_run:
        cmd.append("--dry_run")

    print("\n" + "=" * 70)
    print(f"Executing: {MODEL_DISPLAY_NAMES.get(model_key, model_key)} on {SUBSET_DISPLAY_NAMES.get(norm_sub, norm_sub)}")
    print(f"Command  : {' '.join(cmd)}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Experiment failed for {model_key} on {norm_sub} (code {result.returncode})", file=sys.stderr)
        return False
    return True


def collect_results(output_dir):
    json_pattern = os.path.join(output_dir, "*_results.json")
    json_files = glob.glob(json_pattern)
    records = []

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Warning: Could not read {fpath}: {e}", file=sys.stderr)

    # Sort records: CNN before Wav2Vec2, then sentence, improvisation, both
    def sort_key(item):
        m = item.get("model", "").lower()
        m_order = 0 if "cnn" in m else 1
        s = normalize_subset(item.get("subset", ""))
        s_order = {"sentence": 0, "improvisation": 1, "both": 2}.get(s, 99)
        return (m_order, s_order)

    records.sort(key=sort_key)
    return records


def format_percentage(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def generate_report(output_dir):
    records = collect_results(output_dir)
    if not records:
        print(f"\nNo result JSON files found in: {output_dir}")
        return

    # 1. Main Metrics Table
    headers = ["Model", "Subset", "Accuracy", "Precision", "Recall", "F1-Score"]
    rows = []
    for r in records:
        m_name = MODEL_DISPLAY_NAMES.get(r.get("model", "").lower(), r.get("model", "Unknown"))
        s_name = SUBSET_DISPLAY_NAMES.get(r.get("subset", "").lower(), r.get("subset", "Unknown"))
        acc = format_percentage(r.get("accuracy"))
        prec = format_percentage(r.get("precision"))
        rec = format_percentage(r.get("recall"))
        f1 = format_percentage(r.get("f1_score"))
        rows.append([m_name, s_name, acc, prec, rec, f1])

    # Terminal formatting
    col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    
    def format_row(row_items):
        return " | ".join(f"{item:<{col_widths[i]}}" if i < 2 else f"{item:>{col_widths[i]}}" for i, item in enumerate(row_items))

    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\n" + "=" * 80)
    print("                    EXPERIMENT RESULTS SUMMARY TABLE                    ")
    print("=" * 80)
    print(format_row(headers))
    print(sep_line)
    for row in rows:
        print(format_row(row))
    print("=" * 80)

    # 2. Markdown Table
    md_lines = []
    md_lines.append("# ThaiSER Experiment Results (6-Matrix)")
    md_lines.append("")
    md_lines.append("### Overall Performance Metrics")
    md_lines.append("")
    md_lines.append("| Model | Subset | Accuracy | Precision | Recall | F1-Score |")
    md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for r in rows:
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    md_lines.append("")

    # 3. Per-Class F1 Table if available
    has_per_class = any("per_class" in r and r["per_class"] for r in records)
    if has_per_class:
        classes = ["Angry", "Frustrated", "Happy", "Neutral", "Sad"]
        pc_headers = ["Model", "Subset"] + classes + ["Macro F1"]
        pc_rows = []
        for r in records:
            m_name = MODEL_DISPLAY_NAMES.get(r.get("model", "").lower(), r.get("model", "Unknown"))
            s_name = SUBSET_DISPLAY_NAMES.get(r.get("subset", "").lower(), r.get("subset", "Unknown"))
            pc = r.get("per_class", {})
            f1s = [format_percentage(pc.get(c, {}).get("f1_score")) for c in classes]
            macro_f1 = format_percentage(r.get("f1_score"))
            pc_rows.append([m_name, s_name] + f1s + [macro_f1])

        print("\n" + "=" * 80)
        print("                     PER-CLASS F1-SCORE BREAKDOWN                       ")
        print("=" * 80)
        pc_widths = [max(len(h), max((len(row[i]) for row in pc_rows), default=0)) for i, h in enumerate(pc_headers)]
        def format_pc_row(items):
            return " | ".join(f"{items[i]:<{pc_widths[i]}}" if i < 2 else f"{items[i]:>{pc_widths[i]}}" for i in range(len(items)))
        pc_sep = "-+-".join("-" * pc_widths[i] for i in range(len(pc_headers)))
        print(format_pc_row(pc_headers))
        print(pc_sep)
        for row in pc_rows:
            print(format_pc_row(row))
        print("=" * 80)

        md_lines.append("### Per-Class F1-Score Breakdown")
        md_lines.append("")
        md_lines.append("| Model | Subset | " + " | ".join(classes) + " | Macro F1 |")
        md_lines.append("| :--- | :--- | " + " | ".join([":---:"] * (len(classes) + 1)) + " |")
        for row in pc_rows:
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

    # Output Markdown to file
    md_content = "\n".join(md_lines)
    summary_md_path = os.path.join(output_dir, "experiment_results_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nPresentation-ready Markdown table:\n")
    print(md_content)
    print(f"\nMarkdown summary saved to: {summary_md_path}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.report:
        # Standardize unique models and subsets
        selected_models = []
        for m in args.models:
            norm_m = "wav2vec" if m in ["wav2vec", "wav2vec2"] else "cnn"
            if norm_m not in selected_models:
                selected_models.append(norm_m)

        selected_subsets = []
        for s in args.subsets:
            norm_s = normalize_subset(s)
            if norm_s not in selected_subsets:
                selected_subsets.append(norm_s)

        total_experiments = len(selected_models) * len(selected_subsets)
        print(f"\nStarting Experiment Suite: {total_experiments} total configuration(s)")
        print(f"Models : {[MODEL_DISPLAY_NAMES[m] for m in selected_models]}")
        print(f"Subsets: {[SUBSET_DISPLAY_NAMES[s] for s in selected_subsets]}")
        print(f"Dry Run: {args.dry_run}")

        exp_idx = 1
        for model_type in selected_models:
            for subset in selected_subsets:
                print(f"\n[{exp_idx}/{total_experiments}] Initiating experiment...")
                success = run_experiment(model_type, subset, args)
                if not success:
                    print(f"Aborting subsequent experiments due to error in {model_type}-{subset}.")
                    return
                exp_idx += 1

    # Generate and print report table
    generate_report(args.output_dir)


if __name__ == "__main__":
    main()
