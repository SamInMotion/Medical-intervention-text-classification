"""Apply subsampling support to src/cohen_pipeline.py (in place).

Adds --subsample-n N and --subsample-seed S CLI flags and threads them
through run_cohen_kfold and run_text_mode_comparison. Subsampling is
stratified by label, applied after load_cohen_topic and before text-mode
preparation, so it preserves topic prevalence.

Creates src/cohen_pipeline.py.bak before modifying.
Idempotent: running twice has no effect after first successful application.

Usage:
    # from repo root
    python paper_experiments/patch_cohen_pipeline.py

    # or with explicit path
    python paper_experiments/patch_cohen_pipeline.py src/cohen_pipeline.py
"""

from pathlib import Path
import shutil
import sys


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def patch(target_path):
    target = Path(target_path)
    if not target.exists():
        fail(f"Target file not found: {target}")

    src = target.read_text(encoding="utf-8")

    if "subsample_n" in src and "--subsample-n" in src:
        print(f"[skip] {target} already patched (subsample_n present). Nothing to do.")
        return

    # ----- Edit 1: signature of run_cohen_kfold ---------------------------------
    old_sig_kfold = (
        "def run_cohen_kfold(\n"
        "    topic,\n"
        "    tsv_path,\n"
        "    cache_dir,\n"
        "    email,\n"
        "    api_key=None,\n"
        "    text_mode=\"abstract\",\n"
        "    workflow_id=10,\n"
        "    n_folds=5,\n"
        "    epochs=42,\n"
        "    batch_size=5,\n"
        "):"
    )
    new_sig_kfold = (
        "def run_cohen_kfold(\n"
        "    topic,\n"
        "    tsv_path,\n"
        "    cache_dir,\n"
        "    email,\n"
        "    api_key=None,\n"
        "    text_mode=\"abstract\",\n"
        "    workflow_id=10,\n"
        "    n_folds=5,\n"
        "    epochs=42,\n"
        "    batch_size=5,\n"
        "    subsample_n=None,\n"
        "    subsample_seed=42,\n"
        "):"
    )
    if old_sig_kfold not in src:
        fail("Anchor not found for run_cohen_kfold signature. Aborting.")
    src = src.replace(old_sig_kfold, new_sig_kfold, 1)

    # ----- Edit 2: subsampling block right after load_cohen_topic ---------------
    old_after_load = (
        "    df = load_cohen_topic(\n"
        "        tsv_path=tsv_path,\n"
        "        topic=topic,\n"
        "        cache_dir=cache_dir,\n"
        "        email=email,\n"
        "        api_key=api_key,\n"
        "    )\n"
        "\n"
        "    if len(df) == 0:"
    )
    new_after_load = (
        "    df = load_cohen_topic(\n"
        "        tsv_path=tsv_path,\n"
        "        topic=topic,\n"
        "        cache_dir=cache_dir,\n"
        "        email=email,\n"
        "        api_key=api_key,\n"
        "    )\n"
        "\n"
        "    if subsample_n is not None and len(df) > 0 and subsample_n < len(df):\n"
        "        from sklearn.model_selection import train_test_split\n"
        "        df_sub, _ = train_test_split(\n"
        "            df,\n"
        "            train_size=subsample_n,\n"
        "            random_state=subsample_seed,\n"
        "            stratify=df[\"labels\"],\n"
        "        )\n"
        "        df = df_sub.reset_index(drop=True)\n"
        "        logger.info(\n"
        "            \"Subsampled topic %s to %d records (stratified, subsample_seed=%d)\",\n"
        "            topic, len(df), subsample_seed,\n"
        "        )\n"
        "\n"
        "    if len(df) == 0:"
    )
    if old_after_load not in src:
        fail("Anchor not found for subsampling injection after load_cohen_topic.")
    src = src.replace(old_after_load, new_after_load, 1)

    # ----- Edit 3: signature of run_text_mode_comparison ------------------------
    old_sig_modes = (
        "def run_text_mode_comparison(\n"
        "    topic,\n"
        "    tsv_path,\n"
        "    cache_dir,\n"
        "    email,\n"
        "    api_key=None,\n"
        "    workflow_id=8,\n"
        "    n_folds=5,\n"
        "    epochs=42,\n"
        "    batch_size=5,\n"
        "):"
    )
    new_sig_modes = (
        "def run_text_mode_comparison(\n"
        "    topic,\n"
        "    tsv_path,\n"
        "    cache_dir,\n"
        "    email,\n"
        "    api_key=None,\n"
        "    workflow_id=8,\n"
        "    n_folds=5,\n"
        "    epochs=42,\n"
        "    batch_size=5,\n"
        "    subsample_n=None,\n"
        "    subsample_seed=42,\n"
        "):"
    )
    if old_sig_modes not in src:
        fail("Anchor not found for run_text_mode_comparison signature.")
    src = src.replace(old_sig_modes, new_sig_modes, 1)

    # ----- Edit 4: forward subsample args inside run_text_mode_comparison loop --
    old_inner_call = (
        "    for mode in modes:\n"
        "        result = run_cohen_kfold(\n"
        "            topic=topic,\n"
        "            tsv_path=tsv_path,\n"
        "            cache_dir=cache_dir,\n"
        "            email=email,\n"
        "            api_key=api_key,\n"
        "            text_mode=mode,\n"
        "            workflow_id=workflow_id,\n"
        "            n_folds=n_folds,\n"
        "            epochs=epochs,\n"
        "            batch_size=batch_size,\n"
        "        )"
    )
    new_inner_call = (
        "    for mode in modes:\n"
        "        result = run_cohen_kfold(\n"
        "            topic=topic,\n"
        "            tsv_path=tsv_path,\n"
        "            cache_dir=cache_dir,\n"
        "            email=email,\n"
        "            api_key=api_key,\n"
        "            text_mode=mode,\n"
        "            workflow_id=workflow_id,\n"
        "            n_folds=n_folds,\n"
        "            epochs=epochs,\n"
        "            batch_size=batch_size,\n"
        "            subsample_n=subsample_n,\n"
        "            subsample_seed=subsample_seed,\n"
        "        )"
    )
    if old_inner_call not in src:
        fail("Anchor not found for inner call inside run_text_mode_comparison.")
    src = src.replace(old_inner_call, new_inner_call, 1)

    # ----- Edit 5: argparse additions -------------------------------------------
    old_argparse_anchor = (
        "    parser.add_argument(\n"
        "        \"--output-file\", type=str, default=None,\n"
        "        help=\"Save output to file (in addition to console)\",\n"
        "    )"
    )
    new_argparse_anchor = (
        "    parser.add_argument(\n"
        "        \"--output-file\", type=str, default=None,\n"
        "        help=\"Save output to file (in addition to console)\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--subsample-n\", type=int, default=None,\n"
        "        help=\"Stratified subsample to N records before splitting. \"\n"
        "             \"Use with --subsample-seed for reproducibility.\",\n"
        "    )\n"
        "    parser.add_argument(\n"
        "        \"--subsample-seed\", type=int, default=42,\n"
        "        help=\"Seed for stratified subsample (default: 42)\",\n"
        "    )"
    )
    if old_argparse_anchor not in src:
        fail("Anchor not found for argparse output-file block.")
    src = src.replace(old_argparse_anchor, new_argparse_anchor, 1)

    # ----- Edit 6: pass args from CLI to both entry points ---------------------
    old_cli_modes = (
        "    if args.compare_text_modes:\n"
        "        run_text_mode_comparison(\n"
        "            topic=args.topic,\n"
        "            tsv_path=args.tsv_path,\n"
        "            cache_dir=args.cache_dir,\n"
        "            email=args.email,\n"
        "            api_key=args.api_key,\n"
        "            workflow_id=args.workflow,\n"
        "            n_folds=args.kfold,\n"
        "            epochs=args.epochs,\n"
        "            batch_size=args.batch_size,\n"
        "        )"
    )
    new_cli_modes = (
        "    if args.compare_text_modes:\n"
        "        run_text_mode_comparison(\n"
        "            topic=args.topic,\n"
        "            tsv_path=args.tsv_path,\n"
        "            cache_dir=args.cache_dir,\n"
        "            email=args.email,\n"
        "            api_key=args.api_key,\n"
        "            workflow_id=args.workflow,\n"
        "            n_folds=args.kfold,\n"
        "            epochs=args.epochs,\n"
        "            batch_size=args.batch_size,\n"
        "            subsample_n=args.subsample_n,\n"
        "            subsample_seed=args.subsample_seed,\n"
        "        )"
    )
    if old_cli_modes not in src:
        fail("Anchor not found for CLI compare_text_modes dispatch.")
    src = src.replace(old_cli_modes, new_cli_modes, 1)

    old_cli_kfold = (
        "    else:\n"
        "        run_cohen_kfold(\n"
        "            topic=args.topic,\n"
        "            tsv_path=args.tsv_path,\n"
        "            cache_dir=args.cache_dir,\n"
        "            email=args.email,\n"
        "            api_key=args.api_key,\n"
        "            text_mode=args.text_mode,\n"
        "            workflow_id=args.workflow,\n"
        "            n_folds=args.kfold,\n"
        "            epochs=args.epochs,\n"
        "            batch_size=args.batch_size,\n"
        "        )"
    )
    new_cli_kfold = (
        "    else:\n"
        "        run_cohen_kfold(\n"
        "            topic=args.topic,\n"
        "            tsv_path=args.tsv_path,\n"
        "            cache_dir=args.cache_dir,\n"
        "            email=args.email,\n"
        "            api_key=args.api_key,\n"
        "            text_mode=args.text_mode,\n"
        "            workflow_id=args.workflow,\n"
        "            n_folds=args.kfold,\n"
        "            epochs=args.epochs,\n"
        "            batch_size=args.batch_size,\n"
        "            subsample_n=args.subsample_n,\n"
        "            subsample_seed=args.subsample_seed,\n"
        "        )"
    )
    if old_cli_kfold not in src:
        fail("Anchor not found for CLI default dispatch to run_cohen_kfold.")
    src = src.replace(old_cli_kfold, new_cli_kfold, 1)

    # ----- Write -----
    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[backup] {backup}")
    target.write_text(src, encoding="utf-8")
    print(f"[patched] {target}")
    print("")
    print("Six edits applied:")
    print("  1. run_cohen_kfold gained subsample_n, subsample_seed params")
    print("  2. Stratified subsample block inserted after load_cohen_topic")
    print("  3. run_text_mode_comparison gained matching params")
    print("  4. Inner call inside text-mode loop forwards them")
    print("  5. argparse gained --subsample-n, --subsample-seed")
    print("  6. CLI dispatch threads both args through both entry points")
    print("")
    print("Smoke test (under 2 minutes):")
    print("  python -m src.cohen_pipeline --topic Statins --email YOUR_EMAIL \\")
    print("      --compare-text-modes --subsample-n 200 --subsample-seed 1 \\")
    print("      --kfold 2 --epochs 5 \\")
    print("      --output-file paper_experiments/smoke_test.txt")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "src/cohen_pipeline.py"
    patch(path)
