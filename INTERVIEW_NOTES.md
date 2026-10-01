# NetEval Interview Notes

## 30-second explanation

NetEval is a reliability lab for network AI agents. The LLM chooses what to investigate, but deterministic Python tools backed by NetworkX decide the actual network facts. I capture every tool call, argument, result, and final response, then score the whole trajectory against a fixed evaluation dataset. That exposed failures a final-answer-only metric would miss, such as choosing the wrong tool, stopping early, or making unsupported policy claims. A hardened V2 improved measured task success from 61.11% to 88.89%.

## 90-second explanation

I built NetEval around a simple principle: the model should decide the investigation strategy, while deterministic code should establish truth. The synthetic enterprise network contains devices, services, state, zones, links, ACLs, and a default-deny policy. An NVIDIA-hosted Nemotron model can call five tools for reachability, paths, ACLs, device state, and blocking hops.

Each run produces a structured trajectory. A deterministic evaluator scores tool selection, argument accuracy, trajectory quality, grounding, final-answer correctness, and efficiency across 18 cases. It also attaches specific failure types, so improvements can target actual behavior rather than vague prompt quality.

V1 was persuasive but inconsistent. It sometimes called a state tool for a path question, inferred reachability from a physical route, or answered only half of a compound request. V2 improved the prompt and tool descriptions and added a lightweight evidence-completeness gate. The gate derives requirements from the question and refuses a final answer when an explicit evidence class is missing, while leaving the next action to the model. The hardened benchmark reached 88.89% task success and 0.9756 average score, but repeated regression runs were 5/6 and 6/6, so I present the remaining stochasticity honestly.

## Technical deep-dive talking points

- **Ground truth:** `data/network.json` is loaded into an undirected NetworkX graph. Device metadata remains on nodes; ACLs and policy remain explicit data rather than model context.
- **Tools:** the five tools expose focused, JSON-compatible results. They contain no LLM calls and do not guess.
- **Agent loop:** the OpenAI SDK calls NVIDIA's compatible endpoint. The loop supports multiple tool calls, bounded rounds, explicit dispatch, structured errors, token totals, latency, and trajectory persistence.
- **Versioning:** V1 stays intact. V2 changes agent guidance and orchestration guardrails while using the same model, data, deterministic tools, and evaluator.
- **Evaluation:** 18 cases cover simple reachability, paths, state, ACLs, root cause, and multi-step diagnosis. Scores are deterministic and weighted across six components.
- **Observability:** trajectories preserve exact tool order, inputs, outputs, completion state, errors, latency, tokens, and V2 evidence-gate metadata.
- **Regression:** fixed V1 failures become regression cases with minimum scores and required-tool checks, not just snapshots of answer text.
- **Fair comparison:** V1 and V2 use the same network, model, tools, and cases. Historical result files remain preserved.

## Important design decisions

### Keep network truth outside the LLM

I did not put the full network into the prompt or ask the model to simulate policy. The model queries small deterministic interfaces. This limits hallucination, keeps facts auditable, and makes failures attributable to reasoning or tool use rather than an ambiguous source of truth.

### Evaluate trajectories, not only answers

Two agents can produce the same sentence through very different processes. One may use the correct evidence; another may guess. Trajectory evaluation detects wrong tools, wrong arguments, skipped steps, excessive calls, and unsupported conclusions even when the final wording looks plausible.

### Accept multiple valid investigations

Multi-step cases define required evidence and reasonable budgets without forcing one universal tool order. For example, a diagnosis can sometimes begin with reachability or a blocking-hop tool. The evaluator is strict about evidence while allowing valid strategy variation.

### Make failures explicit

The taxonomy turns a low score into an actionable diagnosis: `wrong_entity`, `premature_answer`, `unsupported_claim`, and similar labels point to different remedies. That made V2 improvements evidence-driven.

### Preserve historical measurements

I kept V1 and earlier V2 artifacts instead of replacing disappointing runs. That makes the improvement story reproducible and prevents later tuning from rewriting history.

## Tradeoffs

- Deterministic scoring is reproducible, but string-based matching cannot fully judge explanation quality or nuanced paraphrases.
- A small synthetic graph enables controlled diagnosis, but it omits real routing protocols, dynamic state, and vendor-specific policy behavior.
- More detailed V2 guidance improved reliability, but increased token use.
- The evidence gate prevents some premature answers, but its intent detection uses lightweight heuristics and can miss unusual phrasing.
- Bounded tool rounds prevent runaway loops, but a model that needs one more step can end in a controlled failure.
- A single model makes the experiment focused, but does not show how the architecture generalizes across providers.

## What failed and what I learned

V1 failed in ways that looked reasonable at first glance. For a path question it repeatedly checked device state. In another case it saw a path and described communication as allowed. Compound questions exposed another weakness: the model could solve one clause and ignore the other.

The lesson was to improve the contract between intent, tools, and evidence rather than add network facts to the prompt. Clear tool boundaries fixed many selection errors. Separating topology from reachability reduced unsupported inference. Tracking evidence requirements made incomplete answers visible at runtime.

The regression stability experiment also mattered. One run passed 5/6 and the next passed 6/6 without code changes. Deterministic tools and evaluation do not make the model itself deterministic, so reliability claims should include repeated runs and distributions rather than one best trajectory.

## Why deterministic tools instead of LLM-generated network truth

Network decisions can affect availability and security. A fluent model-generated claim is not enough. Deterministic tools provide inspectable inputs and outputs, consistently apply topology, state, ACL, and default-policy rules, and can be tested without an API. This architecture also makes the model replaceable: a different agent can use the same truth layer and evaluation suite.

## Why trajectory evaluation instead of final-answer-only evaluation

Final-answer grading misses lucky guesses and hidden operational problems. The trajectory shows whether the agent selected the right capability, targeted the requested entities, gathered enough evidence, avoided unnecessary work, and interpreted the evidence correctly. That is closer to what matters when an agent performs real troubleshooting.

## Why regression coverage matters

Prompt and context changes can fix one behavior while breaking another. Regression cases preserve meaningful prior failures with their original V1 metadata, required tools, and minimum scores. They provide a focused release gate and a historical explanation of why each case exists.

## Why the evidence gate was added

Prompt guidance improved average behavior but did not eliminate run-to-run premature completion. The evidence gate adds a small deterministic check before V2 accepts a final response. It asks: did the tools establish every evidence class implied by the user's question? If not, it tells the model what evidence is missing and lets the model choose the next tool. It improves completeness without embedding case IDs, hard-coded answers, or automatic tool execution.

## Results to remember

- Dataset: 18 evaluation cases in six categories.
- V1: 61.11% task success, 0.7534 overall score.
- Hardened V2: 88.89% task success, 0.9756 overall score.
- Path analysis: 33.3% in V1 to 100% in hardened V2.
- Average tool calls: 1.833 to 1.389.
- Average total tokens: 3,643.06 to 6,198.33.
- Regression stability experiment: 5/6, then 6/6.
- Repeated targeted cases: `multi_02` 3/3 and `multi_03` 3/3.
