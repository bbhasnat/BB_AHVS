# Deep Analysis: AHVS × Knowledge Distillation

## The Question

Are AHVS and the KD system redundant, complementary, or something more powerful when combined?

---

## 1. What Each System Actually Does

**KD System** — a *linear manufacturing pipeline*:
```
Task Spec → Prompts → Data → Best Prompt → Annotations → Validation → Trained Model
```
It answers: *"Given a task, how do I create a cheap model that performs like an expensive LLM?"*
It runs **once per task**. It has no memory. Each run is independent.

**AHVS** — a *cyclic learning engine*:
```
Baseline → Context+Lessons → Hypotheses → Selection → Plan → Execution → Report → Verify → (next cycle)
```
It answers: *"Given a metric, how do I systematically improve it through autonomous experimentation?"*
It runs **N cycles**, accumulating lessons. Each cycle is informed by all previous cycles.

---

## 2. Are They Redundant?

**No. They are orthogonal.**

| Dimension | KD | AHVS |
|---|---|---|
| **Topology** | Linear pipeline | Cyclic loop |
| **Memory** | Stateless | Persistent lessons across cycles + projects |
| **Optimization target** | Produce *a* model | Improve *a metric* |
| **Decision-making** | Fixed stages, human-designed | LLM-generated hypotheses, empirically tested |
| **Iteration** | One-shot | Unbounded cycles |
| **Scope** | Model creation | System improvement |

They share zero code, zero architecture, zero purpose overlap. The word "distillation" in KD refers to teacher→student model transfer. The "learning" in AHVS refers to experimental-lesson accumulation. Different knowledge, different mechanisms.

---

## 3. Why Combining Them Is Powerful

The deep insight is this: **KD is a powerful *action* that AHVS can *discover, configure, execute, and iterate on***. And AHVS's lesson system can make KD progressively smarter across runs. Three integration vectors:

### Vector A: AHVS Optimizes the KD Pipeline Itself

The KD pipeline has dozens of tunable decisions:
- Which prompt candidate style works best (concise vs. rubric-driven vs. conservative)?
- What sample size is sufficient for annotation quality?
- DSPy mode vs. standard mode — when does optimization pay off?
- BERT vs. DistilBERT vs. RoBERTa — which architecture for which task type?
- `gpt-4.1-nano` vs `gpt-4.1-mini` — cost/quality tradeoff for fine-tuning?
- What annotation quality threshold prevents training on noisy labels?

Today these are human choices. With AHVS wrapping the KD pipeline:

```
AHVS Cycle 1: "Hypothesis: Using DSPy MIPROv2 for prompt selection
               improves downstream F1 by >3%"
    → Execute KD pipeline with DSPy enabled
    → Measure final model F1
    → Result: +4.2% F1, lesson archived

AHVS Cycle 2: "Hypothesis: RoBERTa outperforms DistilBERT for
               sentiment tasks when training data < 5000 samples"
    → Execute KD pipeline with RoBERTa
    → Measure final model F1
    → Result: +1.8% F1, lesson archived

AHVS Cycle 3: (informed by Cycles 1-2) "Hypothesis: DSPy + RoBERTa
               combined gives compounding improvement"
    → Tests the interaction effect
    → Lessons accumulate...
```

**This turns a static pipeline into a self-improving one.** After 10 cycles, AHVS's evolution store knows *which KD configurations work for which task types*.

### Vector B: KD as a Hypothesis Type in AHVS

When AHVS is optimizing a RAG system and discovers that an LLM call is the bottleneck:

```
AHVS Hypothesis: "Distill the answer_relevance scoring function
                  from GPT-4o into a fine-tuned gpt-4.1-nano model
                  using the KD pipeline. Expected: 10x latency reduction
                  with <2% quality loss."
```

AHVS would:
1. Generate a task spec YAML for the scoring function
2. Invoke the KD pipeline (auto_ml) to produce a distilled model
3. Swap the distilled model into the RAG system
4. Run eval to measure answer_relevance + latency
5. Archive the lesson: "distilling scoring function → -0.8% relevance, +10x speed"

**KD becomes a tool in AHVS's toolkit** — a new hypothesis type alongside `prompt_rewrite`, `code_change`, and `architecture_change`. Call it `knowledge_distillation`.

### Vector C: Cross-System Memory Flywheel

The most powerful integration:

```
┌─────────────────────────────────────────────────────────┐
│                    AHVS EVOLUTION STORE                   │
│                                                          │
│  "DistilBERT works for binary classification < 10k"      │
│  "DSPy+MIPRO adds +3-5% F1 on multi-label tasks"        │
│  "gpt-4.1-nano fine-tune beats BERT when labels > 20k"   │
│  "Prompt candidate C (conservative) best for legal text"  │
│                                                          │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
        ┌──────▼──────┐        ┌──────▼──────┐
        │  AHVS Cycle │        │  KD Pipeline │
        │  (generates  │◄──────►│  (configured │
        │  hypotheses) │        │  by AHVS)    │
        └─────────────┘        └──────────────┘
```

Lessons from KD runs feed AHVS's context loader. AHVS's hypothesis generator uses those lessons to configure KD better. **The memory is the flywheel.**

---

## 4. Current State of Research

Based on the research landscape (2023-2025), the key finding is:

**This combination is a genuine gap in the literature.**

| What Exists | What's Missing |
|---|---|
| One-shot KD pipelines (Distilling Step-by-Step, Orca, Phi) | Iterative, self-improving KD with accumulated lessons |
| Self-improvement loops (STaR, ReST, SPIN) | External-metric-grounded improvement (not self-referential) |
| DSPy compiles expensive → cheap pipelines | Autonomous discovery of *what* to distill |
| AI Scientist runs experiments autonomously | No memory across experiments, no distillation integration |
| RLHF + distillation is standard at labs | But ad-hoc, not hypothesis-driven |

The closest work:
- **DSPy 2.0** can compile an expensive LLM program into a fine-tuned small model, but has no autonomous hypothesis loop
- **The AI Scientist** (Sakana, 2024) runs experiments but has no persistent memory and doesn't do distillation
- **GKD** (Google DeepMind, 2024) bridges KD and RL but is not autonomous
- **Model collapse** (Shumailov et al., Nature 2024) warns that self-distillation without external grounding degrades — AHVS's eval-against-real-metric approach directly addresses this

**The combined system would be novel** because no published system has:
1. Autonomous hypothesis generation about KD pipeline improvements
2. Isolated testing of those hypotheses
3. Metric-driven evaluation (external grounding, preventing collapse)
4. Persistent cross-cycle memory of what worked
5. Cross-project knowledge transfer

### Key Related Papers

| Year | Paper | Relevance |
|------|-------|-----------|
| 2018 | Christiano — "Supervising strong learners by amplifying weak experts" | IDA foundation |
| 2022 | Zelikman et al. — "STaR: Self-Taught Reasoner" | Self-improvement via self-distillation |
| 2022 | Salimans & Ho — "Progressive Distillation for Fast Sampling" | Iterative distillation formalism |
| 2023 | Hsieh et al. — "Distilling Step-by-Step" (ACL 2023) | Rationale-based KD for LLMs |
| 2023 | Rafailov et al. — "DPO" (NeurIPS 2023) | Implicit distillation of preferences |
| 2023 | Tunstall et al. — "Zephyr" | RLHF + distillation pipeline |
| 2023 | Gu et al. — "MiniLLM" | LLM-specific KD losses |
| 2023 | Khattab et al. — "DSPy" | Compound AI + distillation optimization |
| 2024 | Agarwal et al. — "GKD" | On-policy KD bridging KD and RL |
| 2024 | Shumailov et al. — "AI models collapse" (Nature) | Limits of self-distillation loops |
| 2024 | Lu et al. — "The AI Scientist" | Autonomous ML experimentation |
| 2024 | Hosseini et al. — "V-STaR" | Verified self-improvement |
| 2024 | Chen et al. — "SPIN" | Self-play as iterated distillation |
| 2024 | Hu et al. — "ADAS" | Automated agentic system design |

---

## 5. Architecture of the Integrated System

```
                         ┌──────────────────────────┐
                         │     AHVS ORCHESTRATOR     │
                         │  (Cyclic Learning Engine)  │
                         └─────────┬────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐  ┌──────▼──────┐  ┌──────────▼──────────┐
    │  Standard AHVS     │  │  KD-as-Tool │  │  KD Pipeline        │
    │  Hypothesis Types  │  │  Hypothesis │  │  Optimization       │
    │                    │  │  Type       │  │                     │
    │  • prompt_rewrite  │  │             │  │  AHVS wraps KD and  │
    │  • code_change     │  │  "Distill   │  │  tests configs:     │
    │  • config_change   │  │  component  │  │  • prompt strategy  │
    │  • architecture    │  │  X into a   │  │  • model selection  │
    │  • dspy_optimize   │  │  fine-tuned │  │  • DSPy vs standard │
    │                    │  │  model"     │  │  • sample size      │
    └────────────────────┘  └──────┬──────┘  └──────────┬──────────┘
                                   │                     │
                            ┌──────▼─────────────────────▼──────┐
                            │        KD PIPELINE                 │
                            │  Task Spec → Prompts → Annotate    │
                            │  → Validate → Train → Evaluate     │
                            └──────────────┬─────────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │   METRICS   │
                                    │  F1, Acc,   │
                                    │  Latency,   │
                                    │  Cost       │
                                    └──────┬──────┘
                                           │
                            ┌──────────────▼──────────────────┐
                            │     AHVS EVOLUTION STORE         │
                            │  lessons.jsonl (local + global)  │
                            │  Cross-cycle, cross-project      │
                            └─────────────────────────────────┘
```

### Concrete Integration Points

**1. New hypothesis type in AHVS** — add `knowledge_distillation` to `ahvs/executor.py`:
- Execution strategy: invoke `python -m src.auto_ml.main` with generated config
- Success criterion: distilled model metric >= baseline - regression_floor
- Artifacts: trained model, eval report

**2. New AHVS domain pack** — `domain_packs/kd_prompts.yaml`:
- KD-specific hypothesis templates: "try different teacher model", "increase annotation sample", "switch ML backend", "enable DSPy optimization"
- KD-specific skills: prompt_builder, annotator, tml_classifier, oai_tuner

**3. KD task spec generator in AHVS** — when AHVS decides to distill a component:
- Analyze the target function's input/output signature
- Auto-generate a YAML task spec
- Auto-generate sample data from production logs
- Invoke KD pipeline

**4. Shared eval framework** — KD's Data Analyzer + AHVS's eval_command:
- KD validates annotation quality → feeds into AHVS baseline
- AHVS measures end-to-end metric → feeds into KD iteration

---

## 6. The Compounding Benefit

The real power isn't in either system alone — it's in the **compounding effect** across cycles:

| Cycle | What Happens | Knowledge Gained |
|---|---|---|
| 1 | AHVS tests prompt_rewrite on RAG system | "Prompt rewrite gave +2% relevance" |
| 2 | AHVS tests KD: distill scoring function | "Distilled model: -0.5% quality, 10x faster" |
| 3 | AHVS tests KD: distill with DSPy-optimized prompts | "DSPy annotation +3% label quality → +1.5% model F1" |
| 4 | AHVS combines: better prompts + distilled scorer | "+3.5% relevance, 8x faster — compounding gains" |
| 5 | AHVS applies Cycle 3-4 lessons to new project | Cross-project transfer: "DSPy+distillation works for scoring tasks" |

**After N cycles, you have a system that knows:**
- Which components are worth distilling (high cost, stable behavior)
- Which KD configurations work for which task types
- The quality/cost Pareto frontier for your specific domain
- What hypothesis types compound well together

No existing system does this. The closest analogy is Tesla's data engine, but for LLM/ML systems rather than self-driving.

---

## 7. How to Make It More Powerful

Beyond the basic integration, four amplifiers:

### A. Multi-Objective Optimization
Current AHVS optimizes a single metric. The integrated system should optimize a **Pareto frontier**: quality, latency, cost. A distilled model might lose 1% quality but save 90% cost — AHVS should reason about this tradeoff explicitly.

### B. Cascading Distillation
Instead of distilling to a single student, use AHVS to discover **cascading architectures**: fast model handles easy cases, expensive model handles hard cases, with a learned router. AHVS tests different routing thresholds as hypotheses.

### C. Continuous Distillation Monitoring
After deploying a distilled model, AHVS monitors for **drift** — when the student's real-world performance degrades, it triggers a new KD cycle with fresh data. The lesson store tracks *when* re-distillation is needed (e.g., "sentiment models drift after ~30 days on social media data").

### D. Meta-Distillation
After accumulating lessons across many KD runs, **distill the lessons themselves** into a fine-tuned "KD advisor" model that can predict optimal KD configurations without running full experiments. The evolution store becomes training data for a meta-model.

---

## 8. Verdict

**AHVS and KD are not redundant. They are complementary systems that, when integrated, create something qualitatively new: an autonomous, self-improving knowledge distillation engine with persistent memory.**

This sits at an **open gap** in the research landscape. The key novelty is the combination of:
- Hypothesis-driven experimentation (from AHVS)
- Structured model creation (from KD)
- Persistent cross-cycle memory (from AHVS evolution store)
- External metric grounding (prevents model collapse)

The integration is not trivial but architecturally clean — KD becomes a hypothesis type in AHVS, AHVS wraps KD as an optimization target, and the evolution store serves as the flywheel connecting them. The result compounds across cycles in a way neither system can achieve alone.
