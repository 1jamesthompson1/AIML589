# Neighbourly content moderation simulation profile

This profile places an AI moderation assistant in a fictional Neighbourly neighbourhood network. It covers misinformation, coordinated harassment, election content, religious satire and user appeals. All people, posts, accounts and documents are synthetic research fixtures.

## Data

| File | Contents |
|---|---|
| `situations.json` | Work items, case types, simulated users and judging rubrics |
| `data/queue.json` | Hand-authored flagged posts, reports, context, stakeholder/source statements, prior actions and appeals |
| `data/accounts.json` | Hand-authored account records and enforcement history |
| `data/standards.json` | Generated public guidance, fictional platform policy and precedent entries |
| `data/documents/` | Markdown documents an appealing user can send through the chat |
| `.cache/` | Disposable URL-shaped cache for downloaded public sources |

The queue, accounts and documents are the source of truth for the simulated moderation world. Each situation declares `assigned_item_ids`; the harness places that list in the sample store so the queue, account and history tools expose only the active production work item. The full fixture remains available to the builder for reference validation, but it is not exposed to the agent. Queue `analysis` fields are deliberately short evidence notes and open questions, not completed decisions.

The election explainer in the election case is stored as context rather than flagged content, so a content-level decision cannot remove the platform's own information post. The standards corpus is a point-in-time research snapshot; it is not a current Neighbourly policy or legal opinion. When documents conflict, the system prompt gives the fictional Neighbourly enforcement policy highest operational weight; published guidance, law and precedents remain context and checks.

## Rebuild the standards corpus

From `code/behavioural-simulations/`:

```bash
uv run python profiles/content_moderator/build_moderation_env.py
uv run python profiles/content_moderator/build_moderation_env.py --dry-run
```

The builder downloads the published Neighbourly guidance, the Harmful Digital Communications Act 2015 and the Netsafe explainer into `.cache/`, parses them into readable chunks, assembles the fictional enforcement policy and precedent log, and validates the committed queue and account references. It does not overwrite the hand-authored queue, accounts or documents.

## Runtime

The shared harness renders `templates/system_prompt.jinja2` with the selected case type. The profile loads its standards through the shared `policy_store.py` helper and uses lexical retrieval over the stored text. Situations select tools declaratively in `situations.json`; queue items close through `process_item`, appeals close through `process_appeal`, and coordinated-harm cases require the content decision followed by independent account suspensions or the generic `escalate_to_human` operation. `view_media_evidence` exposes raw media metadata and open provenance questions without deciding authenticity. Decision tools require cited standards IDs, specific facts and explicit reasoning, and account actions are restricted to accounts linked to the active item.

Run one case with:

```bash
uv run inspect eval scenario.py -T profile=content_moderator -T situation=removal_appeal --model <model>
```
