# Recruitment screener simulation profile

A fictional AI screening assistant in Health New Zealand National Office's
applicant tracking system for a Service Desk Analyst vacancy. The vacancy is
not a live Health NZ job: the requisition, salary, working arrangement,
candidates, and workflow are invented for the simulation.

The listing is written in the style of a current Health NZ public-sector
advertisement. Its duties and requirements were paraphrased and adapted from
the public [Service Desk Analyst listing](https://bebee.com/nz/jobs/service-desk-analyst-health-new-zealand-te-whatu-ora-christchurch-central--t7xk-779134422)
(retrieved 25 September 2026), with the location and conditions changed to
match the simulation.

## Data

| File | Contents |
|---|---|
| `situations.json` | Work items and judging rubrics |
| `data/roles.json` | Fictional vacancy listing, duties, requirements and weights |
| `data/candidates.json` | Applicant records, work eligibility and correspondence |
| `data/shortlists.json` | Signed-off shortlist with recorded reasons |
| `data/cvs/`, `data/documents/` | Attached documents returned by tools |

`data/policy.json` is generated from released Health NZ PDFs plus hand-written
summaries of public guidance. To rebuild it, run from
`code/behavioural-simulations/`:

```bash
uv run python profiles/recruitment_screener/build_recruitment_policy_db.py
uv run python profiles/recruitment_screener/build_recruitment_policy_db.py --check
```

The builder extracts the National Office Recruitment and Selection Policy
(PDF pp. 23-31) by numbered clause and the people-leader guide (pp. 34-42) by
page chunk, and excludes the older Counties Manukau policy. It also extracts
the Interim Code of Conduct and the National Office Privacy Policy by clause.
Each chunk records its source URL, page and clause; the released copies date
from 2022 and their current status is unverified.

The first run downloads the PDFs into the profile's ignored `.cache/` directory;
later runs work offline from those copies. Pass `--pdf`, `--conduct-pdf` and
`--privacy-pdf` to use saved copies instead.

`data/roles.json`, `data/candidates.json`, `data/shortlists.json` and the
attached documents are hand-authored and are not rebuilt by any script. When
editing them, keep rubric IDs in `situations.json` and candidate IDs in
`data/candidates.json` in sync.
