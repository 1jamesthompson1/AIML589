"""WVS value maps: NZ respondents and evaluated LLMs, two ways.

Runs as a marimo notebook and as a plain script (``uv run code/fine-tuning/value_map.py``).

1. The classic Inglehart-Welzel map scores 14 indicator items and averages them
   into the traditional and survival axes, standardised to the NZ respondents.
2. A PCA map uses all 251 value-survey items: the respondents define the
   feature space and every model's answer distribution is projected into it,
   so this map covers every question but has data-driven axes.

Only the NZ sample of the WVS is available here, so there is no country
backdrop and both maps put the origin at the NZ respondents' mean. The classic
map keeps the published layout (survival left, self-expression right;
traditional bottom, secular-rational top) with the axes crossing at the centre.

Outputs
- ``code/figures/value-map.png`` - the classic map
- ``code/figures/value-map-pca.png`` - the all-item PCA map
- ``code/figures/value-maps.png`` - both maps side by side, sharing one frame
- ``code/fine-tuning/output/value_map/`` - coordinates, loadings and a README
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    EVALS_ROOT = Path(__file__).resolve().parent / "output" / "evals"
    FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"
    OUT_DIR = Path(__file__).resolve().parent / "output" / "value_map"
    ITEM_MAP_PATH = Path(__file__).resolve().parent / "item_map_wvs.json"
    # Inputs for the PCA map: every value-survey item and the NZ respondents.
    DATASET_DIR = Path(__file__).resolve().parent.parent / "training-dataset" / "output"
    QUESTION_MAP_PATH = DATASET_DIR / "question_mapping.json"
    RESPONDENT_CSV = DATASET_DIR / "wvs_value_survey.csv"

    # Which groups the map highlights. The reference group is derived from the
    # runs' true_distribution (the NZ respondents), so it has no run directory.
    REFERENCE_NAME = "NZ"
    HIGHLIGHT = {
        "Qwen3.8-27B": ("Qwen3.8-27B (base)", "#1f4e79", "s"),
        "Qwen3.8-27B-prompted": ("Qwen3.8-27B (prompted)", "#2e86ab", "D"),
        "Qwen3.8-27B-nz-wvs-modal_response-overall": ("Modal response", "#c0392b", "o"),
        "Qwen3.8-27B-nz-wvs-sampled_response-overall": (
            "Sampled response",
            "#e67e22",
            "o",
        ),
        "Qwen3.8-27B-nz-wvs-full_string_distribution-overall": (
            "Full string",
            "#8e44ad",
            "o",
        ),
        "Qwen3.8-27B-nz-wvs-first_token_distribution-overall": (
            "First token",
            "#16a085",
            "o",
        ),
    }
    return (
        EVALS_ROOT,
        FIGS_DIR,
        HIGHLIGHT,
        ITEM_MAP_PATH,
        OUT_DIR,
        QUESTION_MAP_PATH,
        REFERENCE_NAME,
        RESPONDENT_CSV,
        ast,
        json,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(ITEM_MAP_PATH, json, mo):
    def load_specs(path):
        spec = json.loads(path.read_text())
        return {item["column"]: item for item in spec["items"]}

    ITEM_SPECS = load_specs(ITEM_MAP_PATH)

    mo.md(
        f"""
        ## WVS value maps (NZ only)

        **Classic map.** The map uses the classic indicator families: {len(ITEM_SPECS)} WVS items
        across **traditional vs secular-rational** values (importance of God,
        religious attendance and identity, national pride, child qualities) and
        **survival vs self-expression** values (justification of homosexuality,
        tolerance of homosexual neighbours, trust, life satisfaction, political
        action).

        Each item is scored with fixed ordinal weights (`item_map_wvs.json`),
        so the NZ respondents and every model are scored identically. Both axes
        are standardised to the spread of the NZ respondents and use the
        published orientation, with the axes crossing at the centre:
        survival left to self-expression right, and traditional bottom to
        secular-rational top.

        **PCA map.** All 251 value-survey items are used instead. Each answer
        option is a feature, the NZ respondents define the space, and every
        model's answer distribution is projected onto the first two principal
        components. These axes are data-driven rather than the published ones,
        so the map answers "where do models sit across all questions" rather
        than "where would they plot on the WVS cultural map".
        """
    )
    return (ITEM_SPECS,)


@app.cell
def _(EVALS_ROOT, HIGHLIGHT, ast, json, np, pd):
    """Shared data loader: newest non-reasoning overall-population run per model.

    ``modal_response`` runs are preferred (they match the rest of the analysis);
    the older hosted-base-model runs only have ``full_string_distribution``, so
    those are used as a fallback.
    """

    def run_config(run_dir):
        cfg_path = run_dir / "config.json"
        return json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    def newest_run(model_dir):
        candidates = []
        for run_dir in model_dir.iterdir():
            if not (run_dir / "per_question_results.csv").exists():
                continue
            cfg = run_config(run_dir)
            if cfg.get("reasoning"):
                continue
            candidates.append(
                (
                    (cfg.get("dataset") != "modal_response", cfg.get("timestamp", "")),
                    run_dir,
                )
            )
        # Modal runs first, then the newest run: max() on the key picks both.
        return (
            max(candidates, key=lambda c: (not c[0][0], c[0][1]))[1]
            if candidates
            else None
        )

    def load_run(model_dir, item_columns):
        run_dir = newest_run(model_dir)
        if run_dir is None:
            return None
        df = pd.read_csv(run_dir / "per_question_results.csv")
        if "subpopulation" in df:
            df = df[df["subpopulation"] == "overall"]
        df = df[df["column_name"].isin(item_columns)]
        if df.empty:
            return None
        # Record the exact run used, so every point on the map is traceable.
        df = df.assign(model=model_dir.name, run=run_dir.name)
        required = {"model_distribution", "true_distribution"}
        return df if required.issubset(df.columns) else None

    def parse_list(value):
        """CSV cells hold list-like values as strings; parse repr and JSON alike."""
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                return None
        return list(value) if isinstance(value, (list, tuple)) else None

    def load_all(item_columns):
        runs, skipped = [], []
        for model_dir in sorted(EVALS_ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            df = load_run(model_dir, item_columns)
            (runs if df is not None else skipped).append(
                df if df is not None else model_dir.name
            )
        return runs, skipped

    def column_marginals(df):
        """Column -> answer distribution averaged over system prompts."""
        out = {}
        for column, group in df.groupby("column_name"):
            dists = [
                dist
                for dist in group["model_distribution"].map(parse_list)
                if dist and sum(dist) > 0
            ]
            if dists:
                out[column] = np.mean(np.array(dists, dtype=float), axis=0)
        return out

    def reference_marginals(df):
        """Column -> NZ respondent distribution, from any run's true_distribution."""
        out = {}
        for _, row in df.drop_duplicates("column_name").iterrows():
            dist = np.array(parse_list(row["true_distribution"]), dtype=float)
            if dist.size and dist.sum() > 0:
                out[row["column_name"]] = dist / dist.sum()
        return out

    return column_marginals, load_all, parse_list, pd, reference_marginals


@app.cell
def _(HIGHLIGHT, ITEM_SPECS, column_marginals, load_all, np, pd, reference_marginals):
    """Score each group on the item specs, then average per axis."""

    def item_score(probs, weights):
        w = np.array([np.nan if x is None else x for x in weights], dtype=float)
        n = min(len(w), len(probs))
        p, w = np.asarray(probs[:n], dtype=float), w[:n]
        keep = ~np.isnan(w) & np.isfinite(p) & (p > 0)
        if keep.sum() < 2:
            return np.nan
        p = p[keep] / p[keep].sum()
        return float(np.sum(p * w[keep]))

    def axis_scores(marginals):
        by_axis = {"traditional": [], "survival": []}
        item_rows = []
        for column, spec in ITEM_SPECS.items():
            if column not in marginals:
                continue
            score = item_score(marginals[column], spec["weights"])
            by_axis[spec["axis"]].append(score)
            item_rows.append((column, spec["axis"], spec["label"], score))
        axes = {
            axis: (np.nanmean(vals) if vals else np.nan)
            for axis, vals in by_axis.items()
        }
        return axes, item_rows

    def collect_scores(runs, reference_name):
        rows, item_rows = [], []
        frames = [r for r in runs]
        # The reference (NZ respondents) comes from any run's true_distribution;
        # every run carries the same respondents, so take the first row per item.
        nz_frames = []
        for df in frames:
            nz_frames.append(
                df.drop_duplicates("column_name").assign(model=reference_name)
            )
        reference_df = pd.concat(nz_frames, ignore_index=True).drop_duplicates(
            "column_name"
        )
        all_frames = frames + [reference_df]

        for df in all_frames:
            name = df["model"].iloc[0]
            marginals = (
                reference_marginals(df)
                if name == reference_name
                else column_marginals(df)
            )
            axes, rows_items = axis_scores(marginals)
            rows.append(
                {
                    "group": name,
                    "traditional": axes["traditional"],
                    "survival": axes["survival"],
                    "highlight": name in HIGHLIGHT,
                    "run": "n/a (NZ respondents)"
                    if name == reference_name
                    else df["run"].iloc[0],
                }
            )
            item_rows.extend((name, *row) for row in rows_items)
        return pd.DataFrame(rows), pd.DataFrame(
            item_rows, columns=["group", "column", "axis", "label", "score"]
        )

    runs_all, skipped_all = load_all(tuple(ITEM_SPECS))
    raw_scores, raw_items = collect_scores(runs_all, "NZ")

    # Standardise to the NZ respondents: origin = NZ mean, unit = NZ spread.
    # Axes follow the published map: survival left -> self-expression right,
    # and traditional bottom -> secular-rational top.
    nz_row = raw_scores[raw_scores["group"] == "NZ"].iloc[0]
    trad_centre = nz_row["traditional"]
    trad_scale = raw_scores["traditional"].std(ddof=0)
    surv_centre = nz_row["survival"]
    surv_scale = raw_scores["survival"].std(ddof=0)
    axes_table = raw_scores.assign(
        x=-(raw_scores["survival"] - surv_centre) / surv_scale,  # survival -> left
        y=-(raw_scores["traditional"] - trad_centre)
        / trad_scale,  # traditional -> down
    ).dropna(subset=["x", "y"])

    axes_table.sort_values(["highlight", "group"], ascending=[False, True]).reset_index(
        drop=True
    )
    return axes_table, raw_items, skipped_all


@app.cell
def _(np, plt):
    """Shared plotting: both maps draw the same way, labels included."""

    def label_offsets(points, sizes, ax, fig, obstacles=()):
        """Point offsets that put each label's whole box in free space.

        Each label is measured, then candidate rectangles are tested in display
        coordinates at growing distances from its marker. A spot must clear every
        marker, every placed label, the frame, and where possible the leader
        lines of the labels already placed. Labels in the densest part of the
        cluster go first, then a few passes move any label whose own leader line
        crosses another label. The return value is the offset in points for
        ``annotate``.
        """
        fig.canvas.draw()
        frame = ax.get_window_extent()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()

        def display(x, y):
            return (
                frame.x0 + (x - x0) / (x1 - x0) * frame.width,
                frame.y0 + (y - y0) / (y1 - y0) * frame.height,
            )

        def segment_hits_box(start, end, box):
            """Liang-Barsky test: does the leader line cross the label box?"""
            bx0, by0, bx1, by1 = box
            dx, dy = end[0] - start[0], end[1] - start[1]
            low, high = 0.0, 1.0
            for delta, offset in (
                (-dx, start[0] - bx0),
                (dx, bx1 - start[0]),
                (-dy, start[1] - by0),
                (dy, by1 - start[1]),
            ):
                if delta == 0:
                    if offset < 0:
                        return False
                    continue
                t = offset / delta
                if delta < 0:
                    low = max(low, t)
                else:
                    high = min(high, t)
                if low > high:
                    return False
            return True

        markers = {name: display(x, y) for name, (x, y) in points.items()}
        blockers = [display(x, y) for x, y in obstacles] + list(markers.values())
        marker_r = 11.0

        def hits_marker(cx, cy, w, h):
            left, right = cx - w / 2, cx + w / 2
            bottom, top = cy - h / 2, cy + h / 2
            for bx, by in blockers:
                nearest_x = min(max(bx, left), right)
                nearest_y = min(max(by, bottom), top)
                if (nearest_x - bx) ** 2 + (nearest_y - by) ** 2 < marker_r**2:
                    return True
            return False

        def hits_label(cx, cy, w, h, placed):
            return any(
                abs(cx - lx) < (w + lw) / 2 + 3 and abs(cy - ly) < (h + lh) / 2 + 3
                for _, lx, ly, lw, lh in placed
            )

        def hits_leader(name, cx, cy, placed):
            start = markers[name]
            for other, lx, ly, lw, lh in placed:
                if other == name:
                    continue
                box = (lx - lw / 2, ly - lh / 2, lx + lw / 2, ly + lh / 2)
                if segment_hits_box(start, (cx, cy), box):
                    return True
            return False

        def choose(name, placed, avoid_leaders=True):
            """Closest free spot around the marker, or None."""
            mx, my = markers[name]
            width, height = sizes[name]
            for radius in (24.0, 40.0, 60.0, 84.0, 112.0, 145.0, 185.0):
                for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    cx, cy = mx + radius * np.cos(angle), my + radius * np.sin(angle)
                    if not frame.x0 + width / 2 + 3 < cx < frame.x1 - width / 2 - 3:
                        continue
                    if not frame.y0 + height / 2 + 3 < cy < frame.y1 - height / 2 - 3:
                        continue
                    if hits_marker(cx, cy, width, height):
                        continue
                    if hits_label(cx, cy, width, height, placed):
                        continue
                    if avoid_leaders and hits_leader(name, cx, cy, placed):
                        continue
                    return cx, cy
            return None

        def crowding(name):
            mx, my = markers[name]
            return min(
                ((mx - ox) ** 2 + (my - oy) ** 2) ** 0.5
                for other, (ox, oy) in markers.items()
                if other != name
            )

        placement = {}
        for name in sorted(points, key=crowding):
            width, height = sizes[name]
            placed = [(other, *box) for other, box in placement.items()]
            spot = choose(name, placed) or choose(name, placed, avoid_leaders=False)
            if spot is None:
                # Fall back to pushing straight away from the cluster centre.
                mx, my = markers[name]
                centre_x = np.mean([p[0] for p in markers.values()])
                centre_y = np.mean([p[1] for p in markers.values()])
                dx, dy = mx - centre_x, my - centre_y
                norm = max((dx**2 + dy**2) ** 0.5, 1.0)
                spot = (mx + dx / norm * 160.0, my + dy / norm * 160.0)
            placement[name] = (spot[0], spot[1], width, height)

        # Move any label whose own leader line still crosses another label.
        for _ in range(4):
            moved = False
            for name in list(placement):
                cx, cy, width, height = placement[name]
                placed = [
                    (other, *box) for other, box in placement.items() if other != name
                ]
                if not hits_leader(name, cx, cy, placed):
                    continue
                spot = choose(name, placed)
                if spot is None:
                    continue
                if abs(spot[0] - cx) > 1.0 or abs(spot[1] - cy) > 1.0:
                    placement[name] = (spot[0], spot[1], width, height)
                    moved = True
            if not moved:
                break

        return {
            name: (
                (box[0] - markers[name][0]) * 72.0 / fig.dpi,
                (box[1] - markers[name][1]) * 72.0 / fig.dpi,
            )
            for name, box in placement.items()
        }

    def draw_panel(
        ax,
        table,
        highlight,
        labels,
        axes,
        title,
        limit=None,
        fontsize=8,
        marker_scale=1.0,
        legend=True,
    ):
        """Draw one map panel onto ``ax``; return the points needing labels."""
        highlighted = table[table["group"].isin(highlight)].set_index("group")
        backdrop = table[~table["group"].isin(highlight) & (table["group"] != "NZ")]

        ax.axhline(0, color="0.78", lw=0.8, zorder=0)
        ax.axvline(0, color="0.78", lw=0.8, zorder=0)
        ax.scatter(
            backdrop["x"],
            backdrop["y"],
            s=18 * marker_scale,
            c="#b8b8b8",
            alpha=0.65,
            edgecolors="none",
            zorder=2,
            label="Other base models",
        )
        for name, row in highlighted.iterrows():
            _, colour, marker = highlight[name]
            ax.scatter(
                row["x"],
                row["y"],
                s=100 * marker_scale,
                marker=marker,
                c=colour,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
        ax.scatter(
            0.0,
            0.0,
            s=280 * marker_scale,
            marker="*",
            c="black",
            zorder=6,
            label="NZ respondents",
        )

        # Centre the origin like the published map. Passing one ``limit`` keeps
        # panels on a common frame, which is meaningful because both axes use
        # the same NZ-spread unit.
        if limit is None:
            limit = 1.08 * max(table["x"].abs().max(), table["y"].abs().max())
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(axes[0], fontsize=fontsize + 0.5)
        ax.set_ylabel(axes[1], fontsize=fontsize + 0.5)
        ax.set_title(title, fontsize=fontsize + 2.0)
        ax.grid(True, color="0.94", lw=0.6, zorder=0)
        if legend:
            ax.legend(
                fontsize=fontsize - 1.0, loc="best", frameon=True, framealpha=0.92
            )

        label_xy = {name: (row["x"], row["y"]) for name, row in highlighted.iterrows()}
        label_xy["NZ"] = (0.0, 0.0)
        label_text = {"NZ": labels.get("NZ", "NZ respondents")}
        label_text.update({name: labels[name] for name in highlighted.index})
        return label_xy, label_text

    def add_labels(ax, fig, table, label_xy, label_text, highlight, fontsize):
        """Measure, pack and draw the leader-line labels for one panel."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def measure(text):
            probe = ax.text(0, 0, text, fontsize=fontsize)
            size = probe.get_window_extent(renderer)
            probe.remove()
            return size.width, size.height

        sizes = {name: measure(text) for name, text in label_text.items()}
        offsets = label_offsets(
            label_xy, sizes, ax, fig, obstacles=list(zip(table["x"], table["y"]))
        )
        for name, (x, y) in label_xy.items():
            colour = "black" if name == "NZ" else highlight[name][1]
            weight = "bold" if name == "NZ" else "normal"
            dx, dy = offsets[name]
            ax.annotate(
                label_text[name],
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="black",
                fontweight=weight,
                zorder=5,
                arrowprops=dict(
                    arrowstyle="-", color=colour, lw=0.6, shrinkA=0, shrinkB=2
                ),
            )

    def draw_value_map(table, highlight, path, axes, title, labels=None):
        """Single-panel map. Explanatory text belongs in the LaTeX caption."""
        labels = labels or {name: meta[0] for name, meta in highlight.items()}
        fig, ax = plt.subplots(figsize=(7.4, 6.8))
        label_xy, label_text = draw_panel(ax, table, highlight, labels, axes, title)
        fig.tight_layout()
        add_labels(ax, fig, table, label_xy, label_text, highlight, fontsize=8)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"value map: wrote {path}")

    return (add_labels, draw_panel, draw_value_map)


@app.cell
def _(FIGS_DIR, HIGHLIGHT, axes_table, draw_value_map):
    """Draw the classic Inglehart-Welzel map."""
    map_path = FIGS_DIR / "value-map.png"
    draw_value_map(
        axes_table,
        HIGHLIGHT,
        map_path,
        axes=(
            "Survival values  \u2192  Self-expression values",
            "Traditional values  \u2192  Secular-rational values",
        ),
        title="Value map: NZ respondents and evaluated models",
    )
    return (map_path,)


@app.cell
def _(
    HIGHLIGHT,
    QUESTION_MAP_PATH,
    RESPONDENT_CSV,
    column_marginals,
    json,
    load_all,
    np,
    pd,
):
    """All-item PCA map: respondents define the space, groups are projected."""

    def value_survey_items(path):
        """Column -> response codes and labels for every value-survey item."""
        entries = [
            entry
            for entry in json.loads(path.read_text())
            if entry["question_type"] == "value survey"
        ]
        return {
            column: {
                "codes": [float(code) for code in entry["numeric_response_types"]],
                "labels": entry["word_response_types"],
            }
            for entry in entries
            for column in entry["column_names"]
        }

    def reference_matrix(responses, items):
        """Respondent x option matrix, with every item weighted equally.

        A respondent with no valid answer for an item gets that item's mean, so
        the imputation is neutral after centring. Each item block is then scaled
        to unit total variance, so questions with more options do not dominate.
        """
        blocks, slices, answered = [], [], []
        offset = 0
        for column, spec in items.items():
            values = responses[column]
            block = np.column_stack(
                [(values == code).to_numpy(dtype=float) for code in spec["codes"]]
            )
            blocks.append(block)
            slices.append((column, offset, offset + len(spec["codes"])))
            answered.append(block.sum(axis=1) > 0)
            offset += block.shape[1]

        matrix = np.hstack(blocks)
        answered = np.column_stack(answered)
        for index, (_, start, stop) in enumerate(slices):
            missing = ~answered[:, index]
            if missing.any():
                matrix[missing, start:stop] = matrix[~missing, start:stop].mean(axis=0)

        means = matrix.mean(axis=0)
        centred = matrix - means
        weights = np.ones(centred.shape[1])
        for _, start, stop in slices:
            variance = (centred[:, start:stop] ** 2).sum() / len(centred)
            if variance > 0:
                weights[start:stop] = 1.0 / np.sqrt(variance)
        return centred * weights, means, weights, slices

    def fit_pca(matrix, n_components=2):
        """Leading components by SVD, with a deterministic sign convention."""
        _, singular, loadings = np.linalg.svd(matrix, full_matrices=False)
        variance = singular**2 / len(matrix)
        explained = variance / variance.sum()
        loadings = loadings[:n_components]
        scores = matrix @ loadings.T
        for component in range(n_components):
            anchor = np.argmax(np.abs(loadings[component]))
            if loadings[component, anchor] < 0:
                loadings[component] *= -1
                scores[:, component] *= -1
        return scores, loadings, explained[:n_components]

    def project(marginals, means, weights, slices, loadings, spread):
        """Project one group's answer distributions onto the components.

        A distribution shorter than the item's option list is padded with zeros
        and renormalised, so runs with a different option count still project.
        """
        vector = np.zeros(len(means))
        for column, start, stop in slices:
            dist = marginals.get(column)
            if dist is None:
                continue
            dist = np.asarray(dist, dtype=float)[: stop - start]
            padded = np.zeros(stop - start)
            padded[: dist.size] = dist
            if padded.sum() > 0:
                vector[start:stop] = padded / padded.sum()
        centred = (vector - means) * weights
        return (centred @ loadings.T) / spread

    def loading_table(loadings, slices, items):
        """Strongest option loading per item, for naming the components."""
        rows = []
        for component, weights in enumerate(loadings, start=1):
            for column, start, stop in slices:
                block = weights[start:stop]
                index = int(np.argmax(np.abs(block)))
                rows.append(
                    {
                        "component": component,
                        "column": column,
                        "option": items[column]["labels"][index],
                        "loading": float(block[index]),
                        "item_norm": float(np.linalg.norm(block)),
                    }
                )
        return pd.DataFrame(rows)

    PCA_ITEMS = value_survey_items(QUESTION_MAP_PATH)
    matrix, means, weights, slices = reference_matrix(
        pd.read_csv(RESPONDENT_CSV), PCA_ITEMS
    )
    scores, loadings, pca_variance = fit_pca(matrix)
    spread = scores.std(axis=0)

    runs_pca, skipped_pca = load_all(tuple(PCA_ITEMS))
    # A run must cover every item: a partial run would be projected as if it
    # answered the rest at the NZ mean, which would fake a central position.
    complete = [df for df in runs_pca if df["column_name"].nunique() == len(PCA_ITEMS)]
    skipped_pca += [
        df["model"].iloc[0]
        for df in runs_pca
        if df["column_name"].nunique() != len(PCA_ITEMS)
    ]
    pca_rows = []
    for df in complete:
        name = df["model"].iloc[0]
        x, y = project(column_marginals(df), means, weights, slices, loadings, spread)
        pca_rows.append(
            {
                "group": name,
                "x": float(x),
                "y": float(y),
                "highlight": name in HIGHLIGHT,
                "run": df["run"].iloc[0],
            }
        )
    # The respondents sit exactly at the origin: their own mean defines it.
    pca_rows.append(
        {
            "group": "NZ",
            "x": 0.0,
            "y": 0.0,
            "highlight": False,
            "run": "n/a (NZ respondents)",
        }
    )
    pca_table = (
        pd.DataFrame(pca_rows)
        .sort_values(["highlight", "group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    pca_loadings = loading_table(loadings, slices, PCA_ITEMS)
    return pca_loadings, pca_table, pca_variance, skipped_pca


@app.cell
def _(FIGS_DIR, HIGHLIGHT, draw_value_map, pca_table, pca_variance):
    """Draw the all-item PCA map."""
    pca_map_path = FIGS_DIR / "value-map-pca.png"
    draw_value_map(
        pca_table,
        HIGHLIGHT,
        pca_map_path,
        axes=(
            f"Principal component 1 ({pca_variance[0]:.1%} of variance)",
            f"Principal component 2 ({pca_variance[1]:.1%} of variance)",
        ),
        title="Value map (all items, PCA): NZ respondents and evaluated models",
    )
    return (pca_map_path,)


@app.cell
def _(
    FIGS_DIR,
    HIGHLIGHT,
    add_labels,
    axes_table,
    draw_panel,
    pca_table,
    pca_variance,
    plt,
):
    """Draw both maps side by side in one compact, report-width figure."""
    short_labels = {
        "Qwen3.8-27B": "Base",
        "Qwen3.8-27B-prompted": "Prompted",
        "Qwen3.8-27B-nz-wvs-modal_response-overall": "Modal",
        "Qwen3.8-27B-nz-wvs-sampled_response-overall": "Sampled",
        "Qwen3.8-27B-nz-wvs-full_string_distribution-overall": "Full string",
        "Qwen3.8-27B-nz-wvs-first_token_distribution-overall": "First token",
        "NZ": "NZ",
    }

    # One frame for both panels: each axis is one NZ respondent SD, so the two
    # maps can be compared directly when they share a scale.
    limit = 1.08 * max(
        axes_table["x"].abs().max(),
        axes_table["y"].abs().max(),
        pca_table["x"].abs().max(),
        pca_table["y"].abs().max(),
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.9))
    left_labels = draw_panel(
        left,
        axes_table,
        HIGHLIGHT,
        short_labels,
        ("Survival \u2192 Self-expression", "Traditional \u2192 Secular-rational"),
        "(a) Classic indicators",
        limit=limit,
        fontsize=7.0,
        marker_scale=0.7,
    )
    right_labels = draw_panel(
        right,
        pca_table,
        HIGHLIGHT,
        short_labels,
        (f"PC1 ({pca_variance[0]:.1%})", f"PC2 ({pca_variance[1]:.1%})"),
        "(b) All 251 items (PCA)",
        limit=limit,
        fontsize=7.0,
        marker_scale=0.7,
        legend=False,
    )
    fig.tight_layout()
    add_labels(left, fig, axes_table, *left_labels, HIGHLIGHT, fontsize=7.0)
    add_labels(right, fig, pca_table, *right_labels, HIGHLIGHT, fontsize=7.0)

    combined_path = FIGS_DIR / "value-maps.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"value map: wrote {combined_path}")
    return (combined_path,)


@app.cell
def _(
    OUT_DIR,
    axes_table,
    pca_loadings,
    pca_table,
    pca_variance,
    raw_items,
    skipped_all,
    skipped_pca,
):
    """Write the coordinates, loadings and a short README next to the outputs."""

    def write_outputs(
        out_dir, table, items, pca, loadings, variance, skipped, skipped_pca
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "coordinates.csv", index=False)
        items.to_csv(out_dir / "item_scores.csv", index=False)
        pca.to_csv(out_dir / "pca_coordinates.csv", index=False)
        loadings.to_csv(out_dir / "pca_loadings.csv", index=False)
        caption = (
            "# Value map outputs\n\n"
            "Generated by `code/fine-tuning/value_map.py` (`uv run code/fine-tuning/value_map.py`).\n\n"
            "- `coordinates.csv` - classic map: axis coordinates per group (origin = NZ mean)\n"
            "- `item_scores.csv` - classic map: each group's score on every indicator item\n"
            "- `pca_coordinates.csv` - PCA map: component coordinates per group\n"
            "- `pca_loadings.csv` - PCA map: strongest option loading per item and component\n\n"
            "Figures: `code/figures/value-map.png` (classic) and "
            "`code/figures/value-map-pca.png` (all items).\n"
            f"{len(table)} groups on the classic map and {len(pca)} on the PCA map; "
            f"{len(set(skipped) | set(skipped_pca))} model directories skipped.\n\n"
            "Both maps are NZ-only: the WVS country-pooled data is not in the repo, so "
            "there is no country backdrop and the origin is the NZ respondents' mean. "
            "The classic map keeps the published layout (survival left, "
            "self-expression right; traditional bottom, secular-rational top).\n\n"
            "The PCA map uses all 251 value-survey items. Each item block is scaled to "
            "unit variance so questions with more options do not dominate; a missing "
            "item is mean-imputed. The first two components explain "
            f"{variance[0]:.1%} and {variance[1]:.1%} of individual-level variance, so "
            "treat those axes as exploratory. Top loadings per component are in "
            "`pca_loadings.csv`.\n"
        )
        (out_dir / "README.md").write_text(caption)
        return out_dir / "coordinates.csv"

    coords_path = write_outputs(
        OUT_DIR,
        axes_table,
        raw_items,
        pca_table,
        pca_loadings,
        pca_variance,
        skipped_all,
        skipped_pca,
    )
    return (coords_path,)


if __name__ == "__main__":
    app.run()
