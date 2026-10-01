# NetEval: 5-Minute Demo Walkthrough

This walkthrough presents the problem, the agent architecture, a simple tool call, a multi-step diagnosis, and the measured reliability improvements. The live commands require a configured NVIDIA API key; the test command is offline.

## 1. Problem — 30 seconds

Network answers from an LLM can be fluent and still be wrong. The model might call the wrong tool, use the wrong device or port, stop before gathering enough evidence, or add a policy claim that no tool result supports.

NetEval evaluates the entire investigation rather than grading only the final sentence.

> The LLM decides what to investigate. Deterministic code decides what is true.

## 2. Architecture — 30 seconds

Open the architecture diagram in [README.md](README.md#architecture).

The NVIDIA-hosted model chooses among five tools. The tools query a deterministic NetworkX-backed network model. Every tool call, argument, result, and final answer becomes a trajectory. The evaluator scores tool choice, arguments, trajectory quality, grounding, answer quality, and efficiency.

V2 adds an evidence-completeness gate between tool results and the final answer. The gate can reject premature completion, but it does not choose or execute tools for the model.

## 3. Simple query — 35 seconds

Ask whether one service can reach its database:

```powershell
python -m src.agent --agent-version v2 "Can orders-api reach orders-db on port 5432?"
```

What to point out:

- `check_reachability` is sufficient for this direct question.
- The model supplies the entities and port.
- Deterministic code returns the path, state-aware reachability, and any blocker.
- The trajectory records the evidence and operational metrics.

## 4. Multi-step troubleshooting — 50 seconds

Run the ACL-backed root-cause scenario:

```powershell
python -m src.agent --agent-version v2 "Why can't admin-net reach orders-db on port 5432?"
```

A concise valid investigation is:

```text
check_reachability(source="admin-net", destination="orders-db", port=5432)
inspect_acl(device="internal-fw", source="admin-net", destination="orders-db", port=5432)
```

The deterministic evidence identifies `internal-fw`, `acl-17`, and `explicit_deny`. The important point is that the model investigates, while Python establishes the network truth.

## 5. V1 failure — 35 seconds

Use the preserved [V1 benchmark](results/benchmark_v1.json) to explain a representative failure. For a pure path question, V1 repeatedly called `get_device_state` instead of `trace_path`. Its measured path-analysis success was 33.3%.

V1 also sometimes treated a topology path as proof that traffic was allowed. That is unsafe because links establish possible hops while device state and policy determine reachability.

To reproduce the full V1 benchmark without overwriting the preserved artifact:

```powershell
python -m src.runner --agent-version v1 --output "$env:TEMP\neteval_benchmark_v1.json"
```

This command performs 18 live API runs and is best left for a longer demo.

## 6. V2 improvement — 35 seconds

V2 added clearer intent-specific tool descriptions, explicit separation of path and reachability evidence, compound-intent guidance, duplicate-call protection, and the evidence gate.

The primary measured [hardened V2 benchmark](results/benchmark_v2_hardened.json) improved task success from **61.11% to 88.89%** and overall score from **0.7534 to 0.9756**. Average tool calls fell from **1.833 to 1.389**, while average total tokens increased from **3,643.06 to 6,198.33**.

## 7. Regression and evidence gate — 45 seconds

For a compound request, V2 derives multiple requirements. A path plus reachability question needs both `trace_path` and `check_reachability`. If the model tries to answer after only the path call, the gate records a trigger and asks the model to continue.

In three targeted runs each, `multi_02` and `multi_03` both passed 3/3. One `multi_02` run triggered the gate and recovered missing path evidence. A later full regression stability check produced 5/6 followed by 6/6, showing that reliability improved without becoming perfectly deterministic.

Run the live six-case V2 regression suite with:

```powershell
python -m src.runner --regression --agent-version v2
```

## 8. Benchmark results — 35 seconds

| Metric | V1 | Hardened V2 |
|---|---:|---:|
| Task success | 61.11% | 88.89% |
| Overall score | 0.7534 | 0.9756 |
| Grounding | 78.15% | 100.00% |
| Tool selection | 72.22% | 94.44% |
| Average tool calls | 1.833 | 1.389 |

To execute a new live V2 benchmark and write results outside the repository:

```powershell
python -m src.runner --agent-version v2 --output "$env:TEMP\neteval_benchmark_v2.json" --compare-v1 --comparison-output "$env:TEMP\neteval_comparison.json"
```

## 9. Limitations — 20 seconds

NetEval currently uses synthetic network data, simplified ACL semantics, lightweight intent heuristics, deterministic string-based scoring, and one primary LLM. Provider variability remains visible in latency, transient errors, and repeat-run stability. It is an evaluation lab, not a production network simulator or digital twin.

For an offline closing check:

```powershell
python -m pytest
```
