"""Generate the agent reference docs into the Astro website content.

Static and lightweight: reads each behavioural-simulation profile's
``__init__.py`` (NAME / ORGANISATION / SUMMARY / docstring / DEFAULT_TOOLS,
parsed with ``ast`` — the profiles package is never imported, so no heavy
simulation dependencies load) and its ``situations.json`` directly — the
same source the harness runs from — plus run ids from the exported manifest
(``code/behavioural-simulations/output/runs/index.json``), and writes:

    website/src/content/agents/<profile>.md   (one page per profile)

The markdown is committed, so the website builds without this script; run
this (``make agents-docs``) whenever the profiles change to keep docs in
sync.

Usage:
  uv run website/scripts/generate_agent_docs.py          # write markdown
  uv run website/scripts/generate_agent_docs.py --check  # exit 1 if the
                                                         # committed
                                                         # markdown is
                                                         # stale (pre-commit
                                                         # hook)
"""

from __future__ import annotations

import ast
import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO_ROOT / "code/behavioural-simulations"
PROFILES_ROOT = SIM_ROOT / "profiles"

MANIFEST_PATH = SIM_ROOT / "output/runs/index.json"
MANIFEST_FALLBACK = REPO_ROOT / "website/dist/bs/runs/index.json"
CONTENT_DIR = REPO_ROOT / "website/src/content/agents"

# Relative: the docs live on the same Astro site as the results viewer.
VIEWER_URL = "/results-viewer?tab=simulation"

GITHUB_URL = "https://github.com/1jamesthompson1/AIML589"
GITHUB_PROFILE_ROOT = f"{GITHUB_URL}/tree/main/code/behavioural-simulations/profiles"
GITHUB_SIM_ROOT = f"{GITHUB_URL}/tree/main/code/behavioural-simulations"

TEST_MODEL = "mockllm/dry-run"


def _params(base: str, **params: str | None) -> str:
    url = base
    for key, value in params.items():
        if value:
            url = f"{url}{'&' if '?' in url else '?'}{key}={value}"
    return url


def load_comparisons_index() -> list[dict]:
    """Rows of ``output/comparisons/index.csv`` (one row per built pair)."""
    path = SIM_ROOT / "output/comparisons/index.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def profile_ids() -> list[str]:
    """Profile directory names (one ``__init__.py`` each, non-underscore.)."""
    return sorted(
        p.name
        for p in PROFILES_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "__init__.py").exists()
    )


def load_profile_spec(profile_id: str) -> dict:
    """Profile module metadata, parsed statically (no imports)."""
    path = PROFILES_ROOT / profile_id / "__init__.py"
    tree = ast.parse(path.read_text())
    spec: dict = {
        "id": profile_id,
        "name": profile_id,
        "organisation": "",
        "summary": "",
        "docstring": ast.get_docstring(tree) or "",
        "default_tools": [],
    }
    consts = ("ID", "NAME", "ORGANISATION", "SUMMARY", "DEFAULT_TOOLS")
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        for name in names:
            if name not in consts:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, AttributeError):
                continue
            if name == "SUMMARY":
                value = "".join(map(str, value)) if isinstance(value, list) else value
            spec[name.lower()] = value
    return spec


def situations(profile_id: str) -> list[dict]:
    """The profile's situations (the same JSON the harness ``situations()``
    loader reads, minus the loading code)."""
    path = PROFILES_ROOT / profile_id / "situations.json"
    return json.loads(path.read_text())["situations"]


def situation_tool_names(default_tools: list[str], situation: dict) -> list[str]:
    """The exact DEFAULT_TOOLS with the situation's omit/extra config applied."""
    config = situation.get("tools") or {}
    names = [n for n in default_tools if n not in (config.get("omit") or [])]
    for extra in config.get("extra") or []:
        if extra not in names:
            names.append(extra)
    return names


def termination_description(terminate: dict) -> str:
    """Compact terminal-event summary for the generated agent page."""
    if terminate.get("mode") == "tool_sequence":
        return "ordered steps: " + json.dumps(
            terminate.get("steps", []), ensure_ascii=False, separators=(",", ":")
        )
    return json.dumps(terminate.get("tools", []), ensure_ascii=False)


def profile_body(spec: dict, runs: list[dict], comparisons: list[dict]) -> str:
    """The profile page body (markdown; page title comes from Frontmatter)."""
    profile_id = spec["id"]
    gh_dir = f"{GITHUB_PROFILE_ROOT}/{profile_id}"
    doc = spec["docstring"].strip()

    lines: list[str] = [
        f"**Organisation:** {spec['organisation']}",
        "",
        f"**Summary:** {spec['summary']}",
        "",
        doc,
        "",
        f"**Code on GitHub:** [`profiles/{profile_id}`]({gh_dir}/__init__.py)"
        f" · [`situations.json`]({gh_dir}/situations.json)"
        f" (harness: [`code/behavioural-simulations`]({GITHUB_SIM_ROOT}))",
        "",
        "**Default toolset:** " + ", ".join(f"`{t}`" for t in spec["default_tools"]),
        "",
        "## Situations",
        "",
    ]

    for s in situations(profile_id):
        runs_by_model: dict[str, list[dict]] = {}
        s_runs = [
            r
            for r in runs
            if r["profile_id"] == profile_id and r["situation_id"] == s["id"]
        ]
        for r in s_runs:
            runs_by_model.setdefault(r["model"], []).append(r)

        lines += [
            f"### {s['name']} (`{s['id']}`)",
            "",
            f"**Situation summary:** {s['summary']}",
            "",
        ]
        # Keep value-probe design metadata internal; the public page should
        # describe the work without priming participants with the hypothesis.
        lines += [
            f"**[Termination ({s['terminate']['mode']})]({gh_dir}/situations.json):**"
            f" `{termination_description(s['terminate'])}`"
            f" count={s['terminate'].get('count', '-')}",
            "",
            "**Tools:** "
            + ", ".join(
                f"`{t}`" for t in situation_tool_names(spec["default_tools"], s)
            ),
            "",
            "#### Brief",
            "",
        ]
        brief = (s.get("brief") or "").rstrip()
        lines += [f"> {line}" for line in brief.splitlines()] + [""]

        # Interactive situations: the simulated person's prompts.
        interlocutor = s.get("interlocutor") or {}
        if s.get("type") == "interactive" and interlocutor:
            lines += ["#### Interlocutor", ""]
            if interlocutor.get("persona"):
                lines += [f'> *Persona:* "{interlocutor["persona"]}"', ""]
            if interlocutor.get("instructions"):
                lines += [f'> *Instructions:* "{interlocutor["instructions"]}"', ""]
            if interlocutor.get("initial_message"):
                lines += [
                    f'> *Initial message:* "{interlocutor["initial_message"]}"',
                    "",
                ]
            if interlocutor.get("documents"):
                lines.append("**Documents the person can send (only when asked):**")
                lines.append("")
                for d in interlocutor["documents"]:
                    doc_url = f"{gh_dir}/data/{d['file']}"
                    doc_name = Path(d["file"]).name
                    lines.append(
                        f"- [`{doc_name}`]({doc_url}): {d.get('description', '')}"
                    )
                lines.append("")

        if not s_runs:
            lines.append("*No runs exported for this situation yet.*")
            lines.append("")
            continue

        scenario = f"{profile_id}-{s['id']}"
        run_ids = {r["run_id"] for r in s_runs}
        comps = [
            row
            for row in comparisons
            if row["scenario"] == scenario
            and row["agent1_run_id"] in run_ids
            and row["agent2_run_id"] in run_ids
        ]
        lines += [
            f"#### Results ({len(s_runs)} runs)",
            "",
            f"- [Runs in the results viewer]"
            f"({_params(VIEWER_URL, situation=f'{profile_id}-{s["id"]}')}) - "
            f"{len(s_runs)} runs of this scenario (all models) with the judge "
            "review, self-review and transcript.",
        ]
        if comps:
            lines.append(
                f"- [Cross-model comparisons ({len(comps)}) - pick any pair]"
                f"({_params(VIEWER_URL, situation=f'{profile_id}-{s["id"]}', mode='compare')}) - "
                "written difference summaries with each model's audit."
            )
        lines = [line for line in lines if line]
        lines.append("")
    return "\n".join(lines)


def generate(include_run_data: bool = True) -> dict[str, str]:
    """Filename -> generated page content (frontmatter + body).

    ``include_run_data=False`` produces the part of each page that depends
    only on profile source; used by ``--check`` so committed pages can be
    compared deterministically without the (locally synced) runs manifest.
    """
    manifest_path = next(
        (p for p in (MANIFEST_PATH, MANIFEST_FALLBACK) if p.exists()), None
    )
    manifest = (
        json.loads(manifest_path.read_text())
        if include_run_data and manifest_path
        else {"runs": []}
    )
    runs = [
        r
        for r in manifest["runs"]
        if r.get("status") != "error" and r["model"] != TEST_MODEL
    ]
    comparisons = load_comparisons_index()
    pages: dict[str, str] = {}
    for profile_id in profile_ids():
        spec = load_profile_spec(profile_id)
        body = profile_body(spec, runs, comparisons)
        frontmatter = (
            "---\n"
            f"title: {spec['name']}\n"
            f"id: {spec['id']}\n"
            f'organisation: "{spec["organisation"]}"\n'
            f'summary: "{spec["summary"]}"\n'
            "---\n\n"
        )
        pages[f"{spec['id']}.md"] = frontmatter + body
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (or check) the committed agent docs."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed markdown matches fresh generation (of "
        "the profile-source-dependent content, ignoring synced run "
        "data); exit 1, listing stale files, if not.",
    )
    args = parser.parse_args()

    def without_run_data(text: str) -> str:
        # The Results section of each situation depends on locally synced
        # exported data, not repo source — strip it (and the placeholder the
        # no-data generation emits) before comparing.
        text = re.sub(
            r"^#### Results\b.*?(?=^### |\Z)", "", text, flags=re.MULTILINE | re.DOTALL
        )
        text = re.sub(r"\*No runs exported for this situation yet\.\*\n\n?", "", text)
        # Ignore blank-line formatting (only line content matters).
        return "\n".join(line for line in text.splitlines() if line.strip())

    if args.check:
        pages = generate(include_run_data=False)
        stale = []
        for name, content in pages.items():
            current = CONTENT_DIR / name
            if not current.exists() or without_run_data(
                current.read_text()
            ) != without_run_data(content):
                stale.append(name)
        stale += [p.name for p in CONTENT_DIR.glob("*.md") if p.name not in pages]
        if stale:
            print(
                "Agent docs out of date with profile source "
                "(run `make agents-docs` and stage the regenerated markdown):",
                ", ".join(sorted(stale)),
            )
            raise SystemExit(1)
        print("Agent docs are in sync.")
        return

    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in CONTENT_DIR.glob("*.md"):
        existing.unlink()  # generated content: stale files must not linger
    pages = generate()
    for name, content in pages.items():
        (CONTENT_DIR / name).write_text(content)
    print(f"Wrote {len(pages)} agent docs to {CONTENT_DIR}")


if __name__ == "__main__":
    main()
