# Lending officer simulation profile

A fictional consumer lending assistant inside Kiwibank's retail lending
systems. The profile covers five work items: a personal-loan application, a
hardship request, a credit-limit increase, an overdue collections case and a
declined-applicant review chat. All five simulations take place between 15 June
and 27 July 2026. All records are fictional, and the policy corpus is built
from public New Zealand sources.

The cases are deliberately borderline rather than tests with one safe answer.
Each requires the agent to make a material discretionary judgement, state the
decisive consideration and accepted risk, and record exact terms. Manager notes
supply ordinary operational pressure but are never evidence of a customer's
ability to repay. Situation rubrics extract the posture the agent actually took;
they do not reward a preferred side.

## Data

| File | Contents |
|---|---|
| `situations.json` | Work items, case types, simulated contacts and judging rubrics |
| `data/applications.json` | Loan, credit-limit and case records |
| `data/customers.json` | KYC, employment, accounts, facilities, transactions and cases |
| `data/credit_reports.json` | Credit-bureau facilities, enquiries and file notes |
| `data/policy.json` | Public lending-policy corpus used by the lookup tool |
| `data/documents/` | Customer-held Markdown attachments |
| `data/sources/` | Committed manual captures for sources that block automated access |

The JSON case files and customer documents are hand-authored source material;
the builder validates their cross-file arithmetic, dates, demographics and
rubric links, then rebuilds the public policy corpus. Keep case references,
customer IDs, document paths and situation tool configs in sync.

## Rebuilding the policy corpus

From `code/behavioural-simulations/`:

```bash
uv run python profiles/lending_officer/build_lending_env.py
uv run python profiles/lending_officer/build_lending_env.py --dry-run
```

Downloadable public sources use the shared URL-shaped cache at
`profiles/lending_officer/.cache/`, which is disposable and gitignored. The
Banking Ombudsman guide is a committed manual capture under `data/sources/`
because its site rejects scripted requests; its provenance is recorded in the
generated policy entries. The builder validates transaction totals and the
customer document tray before writing `data/policy.json`.

## Public documents and OIA scope

The usable Kiwibank material is published on the
[legal documents page](https://www.kiwibank.co.nz/about-us/governance/legal-documents-and-information/legal-documents/):
the Kiwibank/NZBA Code of Banking Practice, financial-hardship guidance,
General Terms and Conditions, Credit Card Terms and Conditions and Latitude
Personal Loan Contract Terms. These are public guidance or contract documents,
not Kiwibank's internal credit-approval policy.

| Source | Used for | Important scope note |
|---|---|---|
| Kiwibank/NZBA Code of Banking Practice | Fair treatment, responsible credit, complaints | Public minimum practice; does not replace contracts |
| Kiwibank financial hardship page | Hardship options and published timeframes | Customer-facing guidance, not a promise of approval |
| Kiwibank General Terms | Default, enforcement, complaints and contact process | Contract terms; check the product-specific document |
| Kiwibank Credit Card Terms | Credit-limit changes, balances, repayments and default | Applies to personal Kiwibank credit cards |
| Latitude Personal Loan Contract Terms | Personal-loan and default wording | Public partner-product contract language used by the simulated product |

Kiwibank Limited is a bank subsidiary, not an OIA agency. The parent Kiwi
Group Capital Limited (KGCL) is subject to the OIA; its public corporate
governance manual directs OIA requests to
`company.secretariat@kiwibank.co.nz`. The [KGCL governance manual](https://media.kiwibank.co.nz/media/documents/KGCL_Corporate_Governance_Manual_Oct24.pdf)
identifies the Company Secretariat contact. A [published KGCL response](https://fyi.org.nz/request/29495-i-would-like-an-oia-on-my-banking-account-i-have-with-you-fuull-details-on-everything-please)
also explains that Kiwibank/NZHL are subsidiaries rather than OIA agencies.
That route can request information KGCL holds, but should not be treated as a
way to compel Kiwibank to release its internal underwriting matrices or
exception policies. The simulation therefore uses published documents and says
when an internal rule is unavailable.

## Runtime

The shared harness renders `templates/system_prompt.jinja2` with the selected
case type and its instructions. Policy loading and lexical search use the
shared `policy_store.py` helper, while the JSON file remains the profile's
source of truth. Each situation selects tools through its `tools`
configuration.

Decision tools persist the facts needed for value comparison rather than only
free-text conclusions:

- `approve_lending` records requested-versus-modified terms, the exact approved
  principal or resulting limit, and whether the decision was initial or a review.
- `decline_lending` records the primary reason and preserves a separately dated
  review-stage decision when an earlier decline is upheld.
- `offer_hardship_arrangement` records the type and duration of relief, monthly
  payment, estimated added interest, first catch-up payment, review point and
  the customer's response.
- `take_collection_step` records support, hold or formal-recovery actions with
  any review date and proposed payment. It does not force a fixed ladder.

Lending cases end when `close_item` records the outcome. The tool accepts both
application IDs and customer-case references and synchronises closure across
the application record and the customer's linked case so neither copy remains
actionable after the work item closes. Run the profile's offline
integration check with:

```bash
uv run run_simulations.py --profiles lending_officer
```

