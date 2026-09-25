# ED guidance captures

These files are readable, dated text captures used by `build_triage_env.py`. They are kept separate from the hand-authored `patients.json` and `department_snapshots.json` fixtures. The generated `protocols.json` records the URL, authority class, jurisdiction, scope and dates for each entry; runtime retrieval applies the simulated work date as an as-of limit. Fetched pages are pinned by `data/source_hashes.json` so an unreviewed source change fails closed.

- `acem_g24.md` — ACEM G24 V6 implementation guidance (ATS assessment, reassessment, documentation and audit).
- ACEM P06 V5 is fetched from the dated policy page by the builder and is represented in `protocols.json` with its authority and review metadata.
- `cchv_te_pae_tiaki_triage.md` — CCHV document 1.989, Te Pae Tiaki ED triage, issue 19 February 2024, pages 2–7.
- `cchv_te_pae_tiaki_overcrowding.md` — CCHV document 1.1560, responsibility/overcrowding/VIS and Duty Nurse Manager/hospital-management escalation, issue 20 February 2025, pages 2–5.
- `healthnz_acute_care.md` — Health New Zealand acute-care, six-hour ED target, access-block and whole-system flow guidance, captured 25 September 2026.
- `healthnz_healthline.md` — current Healthline routing, callback, Māori clinician, interpreter and NZ Relay information, captured 25 September 2026.
- `healthnz_ed_flow.md` and `healthnz_ankle.md` — existing dated Health New Zealand public captures retained by the profile.

The CCHV/Te Pae Tiaki documents are regional references; Te Pae Tiaki is the Wellington ED, so the runtime must not silently treat them as a current Hutt Hospital order. Check the live controlled source and escalate when applicability or a current threshold is unclear.
