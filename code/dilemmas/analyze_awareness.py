"""Analysis of the situational-awareness scan over the dilemma eval logs.

Reads the latest scan under ``output/scans`` (produced by ``sa_scanner.py``,
see the code/dilemmas README for the scan command) and produces:

- the model x dilemma table of mean awareness scores (0-4) plus scan counts,
- a heatmap of the same table, saved to ``output/awareness-heatmap.png``,
- the same rollup by profile (pooled means), saved to
  ``code/figures/awareness-profile-heatmap.png``,
- a signal-type count table (which signs each model showed),
- the highest-scoring responses with their evidence quotes for a quick
  qualitative read.

Run headless from the repository root:

    uv run python code/dilemmas/analyze_awareness.py

or interactively with ``uv run marimo edit code/dilemmas/analyze_awareness.py``.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from inspect_scout import scan_results_df

    HERE = Path(__file__).resolve().parent
    SCANS_DIR = HERE / "output" / "scans"
    OUT_DIR = HERE / "output"
    FIG_DIR = HERE.parent / "figures"
    SCANNER_KEY = "situational_awareness"  # scanner name from sa_scanner.py
    return (
        FIG_DIR,
        HERE,
        OUT_DIR,
        SCANS_DIR,
        SCANNER_KEY,
        json,
        np,
        pd,
        plt,
        scan_results_df,
    )


@app.cell
def _(HERE, SCANS_DIR, SCANNER_KEY, json, scan_results_df):
    # Load the latest scan. The transcript metadata carries the eval sample
    # metadata (profile_id, dilemma_id), so scan results join back to the
    # dilemma table without touching the logs again. Model names are
    # shortened to their final segment (z-ai/glm-5.3-flash -> glm-5.3-flash).
    # Every successful scan is kept: repeated runs of the same dilemma
    # (epochs, spot tests) average into the same cell; the count table shows
    # how much data each cell rests on.

    def latest_scan_dir(scans_dir):
        """The most recently written scan output directory."""
        scan_dirs = sorted(
            scans_dir.glob("scan_id=*"), key=lambda path: path.stat().st_mtime
        )
        if not scan_dirs:
            raise FileNotFoundError(f"no scan_id=* directories in {scans_dir}")
        return scan_dirs[-1]

    def as_mapping(value):
        """Scout result metadata columns arrive as dict or JSON string."""
        if isinstance(value, str):
            return json.loads(value)
        return value if isinstance(value, dict) else {}

    def scan_frame(scans_dir):
        """Scanned rows joined with the eval sample metadata."""
        scan_dir = latest_scan_dir(scans_dir)
        results = scan_results_df(str(scan_dir))
        df = results.scanners[SCANNER_KEY].copy()
        failed = int(df["scan_error"].notna().sum())
        df = df[df["scan_error"].isna()].copy()
        sample_metadata = [
            as_mapping(meta).get("sample_metadata") or {}
            for meta in df["transcript_metadata"]
        ]
        df["profile_id"] = [meta.get("profile_id", "") for meta in sample_metadata]
        df["dilemma_id"] = [meta.get("dilemma_id", "") for meta in sample_metadata]
        df["model"] = [model.split("/")[-1] for model in df["transcript_model"]]
        df["signals"] = df["metadata"].map(
            lambda meta: as_mapping(meta).get("signals", [])
        )
        df["quote"] = df["metadata"].map(lambda meta: as_mapping(meta).get("quote", ""))
        return scan_dir, failed, df

    scan_dir, scan_failed, scan_df = scan_frame(SCANS_DIR)
    print(f"scan {scan_dir.name}: {len(scan_df)} transcripts, {scan_failed} failed")
    return scan_failed, scan_df


@app.cell
def _(HERE, json, scan_df):
    # Profile order as in profiles/, so heatmap columns follow the dilemma
    # listing rather than the alphabet.

    def load_profile_order():
        """Profile ids in profiles/ file order. The stem and the profile id
        inside the JSON differ for some profiles (clergy_advisor.json holds
        profile 'parish_advisor'), and dilemmas are keyed by the latter."""
        return [
            json.loads(path.read_text(encoding="utf-8"))["profile"]
            for path in sorted((HERE / "profiles").glob("*.json"))
        ]

    def sample_index_by_dilemma(df):
        """Map dilemma_id -> its sample index (the number in the sample id)."""
        indexes = (
            df.assign(
                sample_index=[
                    str(task_id).rsplit("-", 1)[-1]
                    for task_id in df["transcript_task_id"]
                ]
            )
            .drop_duplicates("dilemma_id")
            .set_index("dilemma_id")["sample_index"]
            .astype(int)
        )
        return indexes.to_dict()

    profile_order = load_profile_order()
    sample_index = sample_index_by_dilemma(scan_df)
    return profile_order, sample_index


@app.cell
def _(OUT_DIR, profile_order, pd, sample_index, scan_df):
    # Model x dilemma mean awareness (0-4 per scanned response). A cell is
    # empty when that (model, dilemma) pair was never run in the logs. Rows
    # are sorted by each model's overall mean (most aware on top). The
    # detailed per-cell numbers are printed below the tables when run
    # headless; scan parquet retains every underlying result.

    def model_dilemma_table(df):
        valid = df.dropna(subset=["value"])
        pivot = valid.pivot_table(
            index="model", columns="dilemma_id", values="value", aggfunc="mean"
        )
        counts = valid.pivot_table(
            index="model", columns="dilemma_id", values="value", aggfunc="size"
        )
        return pivot, counts

    def dilemma_column_order(dilemmas):
        """Live-profile dilemmas first, grouped by profile order; defunct
        profiles after. The dilemma -> profile mapping comes from the scan
        rows themselves, not name matching."""

        def key(dilemma):
            profile = dilemma_profile.get(dilemma, "")
            live = profile in profile_order
            return (
                0 if live else 1,
                profile_order.index(profile) if live else 0,
                profile if not live else "",
                sample_index.get(dilemma, 999),
                dilemma,
            )

        return sorted(dilemmas, key=key)

    def dilemma_profile_mapping(df):
        """Dilemma_id -> profile_id. Dilemmas were re-run under different
        profiles across history (profile merges), so prefer a current
        profile over the retired one they may have started under."""
        groups = df.groupby("dilemma_id")["profile_id"].agg(
            lambda profiles: sorted(set(profiles))
        )
        mapping = {}
        for dilemma, profiles in groups.items():
            live = [profile for profile in profiles if profile in profile_order]
            mapping[dilemma] = live[0] if live else profiles[0]
        return mapping

    def oriented_tables(df):
        """Mean/count tables, columns profile-grouped, rows by overall mean."""
        raw_means, raw_counts = model_dilemma_table(df)
        ordered = raw_means.reindex(
            columns=dilemma_column_order(list(raw_means.columns))
        )
        row_order = ordered.mean(axis=1).sort_values(ascending=False).index
        means = ordered.reindex(index=row_order)
        counts = raw_counts.reindex(index=means.index, columns=means.columns)
        return means, counts

    dilemma_profile = dilemma_profile_mapping(scan_df)
    mean_table, count_table = oriented_tables(scan_df)

    print("overall mean awareness by model (most aware first):")
    for model, overall in mean_table.mean(axis=1).round(2).items():
        n = int(count_table.loc[model].sum())
        print(f"  {model:>22}  mean={overall}  n={n}")
    print("per-dilemma cells:")
    print(mean_table.round(2).to_string())
    return count_table, mean_table


@app.cell
def _(count_table, mean_table, np, OUT_DIR, plt):
    # Heatmap of mean awareness per model x dilemma; grey cells have no data
    # in the logs. Saved to output/awareness-heatmap.png and shown inline.

    def plot_heatmap(means, counts, out_path, title, xlabel):
        grid = means.to_numpy(dtype=float)
        blocked = np.ma.masked_invalid(grid)
        fig, ax = plt.subplots(
            figsize=(
                max(4.0, 0.3 * len(means.columns) + 2.8),
                max(2.2, 0.45 * len(means.index) + 1.4),
            )
        )
        cmap = plt.colormaps["YlOrRd"].copy()
        cmap.set_bad("#e8e8e8")
        image = ax.imshow(blocked, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        fig.colorbar(image, ax=ax, label="mean awareness score (0-4)")
        ax.set_xticks(range(len(means.columns)))
        ax.set_xticklabels(means.columns, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(means.index)))
        ax.set_yticklabels(means.index, fontsize=8)
        n_table = counts.notna().to_numpy()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if n_table[row, col]:
                    ax.text(
                        col,
                        row,
                        f"{grid[row, col]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                    )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("model")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    heatmap_path = plot_heatmap(
        mean_table,
        count_table,
        OUT_DIR / "awareness-heatmap.png",
        "Situational-awareness scan: model x dilemma",
        "dilemma",
    )
    print(f"wrote {heatmap_path}")
    return heatmap_path, plot_heatmap


@app.cell
def _(FIG_DIR, mean_table, plot_heatmap, profile_order, scan_df):
    # Profile rollup: mean awareness per model x profile, pooling every
    # scanned response run under that profile (weight by responses, not by
    # dilemma cells). Columns: live profiles in profiles/ file order, then
    # retired profiles alphabetically. Reported in the console and plotted
    # to code/figures/awareness-profile-heatmap.png for the report; the
    # per-dilemma heatmap stays in output/.

    def profile_column_order(profiles):
        live = [profile for profile in profile_order if profile in profiles]
        retired = sorted(
            profile for profile in profiles if profile not in profile_order
        )
        return live + retired

    def profile_tables(df):
        valid = df.dropna(subset=["value"])
        means = valid.pivot_table(
            index="model", columns="profile_id", values="value", aggfunc="mean"
        )
        counts = valid.pivot_table(
            index="model", columns="profile_id", values="value", aggfunc="size"
        )
        return means, counts

    def oriented_profile_tables(df):
        raw_means, raw_counts = profile_tables(df)
        ordered = raw_means.reindex(
            columns=profile_column_order(list(raw_means.columns))
        )
        means = ordered.reindex(index=mean_table.index)
        counts = raw_counts.reindex(index=means.index, columns=means.columns)
        return means, counts

    profile_means, profile_counts = oriented_profile_tables(scan_df)

    print("model x profile pooled means (rows by overall mean):")
    print(profile_means.round(2).to_string())
    profile_heatmap_path = plot_heatmap(
        profile_means,
        profile_counts,
        FIG_DIR / "awareness-profile-heatmap.png",
        "Situational-awareness scan: model x profile (pooled means)",
        "profile",
    )
    print(f"wrote {profile_heatmap_path}")
    return (profile_heatmap_path,)


@app.cell
def _(scan_df):
    # Which awareness signal types showed up, per model (model x signal
    # counts). Reported in the console only; the scan parquet holds the
    # per-response signal lists.

    def signal_counts(df):
        exploded = df.explode("signals").dropna(subset=["signals"])
        return (
            exploded.groupby(["model", "signals"], as_index=False)
            .size()
            .rename(columns={"signals": "signal", "size": "n"})
            .sort_values(["model", "n"], ascending=[True, False])
        )

    signal_table = signal_counts(scan_df)
    signal_wide = signal_table.pivot_table(
        index="model", columns="signal", values="n", aggfunc="sum", fill_value=0
    )
    print(f"{int(signal_table['n'].sum())} signals across models:")
    print(signal_wide.to_string())
    return (signal_table,)


@app.cell
def _(scan_df):
    # The highest-scoring responses with their evidence quotes, for a quick
    # qualitative check on what the scanner is reacting to.

    def top_examples(df, n=20):
        columns = ["model", "profile_id", "dilemma_id", "value", "quote"]
        return df.sort_values("value", ascending=False).head(n)[columns]

    top_examples_df = top_examples(scan_df)
    for row in top_examples_df.itertuples(index=False):
        quote = (row.quote or "").replace("\n", " ")
        print(f"{row.value:>3}  {row.model:>20}  {row.dilemma_id:<34}  {quote[:110]}")
    return (top_examples_df,)


if __name__ == "__main__":
    app.run()
