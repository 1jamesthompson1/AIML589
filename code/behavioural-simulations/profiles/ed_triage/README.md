# Emergency department triage simulation profile

This profile places an AI triage assistant in a fictional ED at Hutt Hospital. It covers two waiting-room queue reviews, monitored-bed allocation, an automated waiting-room checkup and an after-hours nurse-line call. The clinical records are synthetic and are not medical advice.

## Data

| File | Contents |
|---|---|
| `situations.json` | Work items, case types, assigned patient IDs, operational snapshot IDs, callers and judging rubrics |
| `data/patients.json` | Hand-authored patient records, observations, triage history, monitored-bed capabilities and the caller record |
| `data/department_snapshots.json` | Point-in-time fictional snapshots: staffing/roster, demand and arrivals, treatment/downstream capacity, flow and ATS breaches, escalation/VIS, alternatives, communications and provenance |
| `data/protocols.json` | Generated, source-labelled clinical and operational guidance used by `lookup_triage_protocol` and `lookup_operational_guidance` |
| `data/sources/` | Committed text captures for sources that block automated fetching or are best pinned to a dated PDF/page capture |
| `data/source_hashes.json` | SHA-256 pins for fetched source snapshots; a changed download fails closed until reviewed |
| `.cache/` | Disposable URL-shaped cache for downloaded public sources |

`patients.json` and `department_snapshots.json` are hand-authored fixtures and are the source of truth for the simulated ED. Guidance captures are point-in-time references, not a current Hutt Hospital order. The runtime labels each source with its authority class, jurisdiction, scope and dates, applies the simulated work date as an as-of limit, and prioritises NZ/Australasian sources for local actions; the CCHV/Te Pae Tiaki material is a regional reference rather than an automatic Hutt order, and unavailable local thresholds must trigger escalation rather than invention.

## Rebuild the guidance corpus

From `code/behavioural-simulations/`:

```bash
uv run python profiles/ed_triage/build_triage_env.py
uv run python profiles/ed_triage/build_triage_env.py --dry-run
```

The builder downloads public pages into `.cache/`, parses or reads the dated captures, and writes `protocols.json`. It validates but never overwrites `patients.json` or `department_snapshots.json`. The committed captures include ACEM P06/G24, CCHV/Te Pae Tiaki triage and overcrowding responsibility material, Health New Zealand acute-care/ED/Healthline guidance, and the existing clinical red-flag sources.

## Runtime

The shared harness renders `templates/system_prompt.jinja2` with the selected case type. `scenario_setup` stores the situation's assigned patient IDs and department snapshot ID in the sample store, then copies the data directory into the sandbox. Queue tools expose only assigned patients, and `view_ed_operations` exposes the selected snapshot's useful operational fields. `record_queue_order` requires a category for every assigned patient and records the complete order. Bed allocation requires an eligible waiting patient, a compatible free monitored bed and a clinical basis; escalation is a valid alternative. A residual-risk handover uses the generic `write_clinical_note` with its handover fields. `process_call` requires staged spoken history, advice, safety-net and goodbye messages; its advice and safety-net fields must quote the spoken text, and it requires confirmation before closing.

Queue situations close through the generic `tool_sequence`; monitored-bed cases allocate or escalate and then require a separate residual-risk note; checkups require a current observation before `automated_checkup_decision`; phone calls close through `process_call`.

Run one case with:

```bash
uv run inspect eval scenario.py -T profile=ed_triage -T situation=phone_triage_child --model <model>
```

