# AHVS Brainstorm — Pre-Genesis Design Exploration

Brainstorm is the first stage of the AHVS pipeline. It answers **"what should we build and why?"** before genesis answers **"how do we scaffold it?"**

```
brainstorm → genesis → onboarding → ahvs cycles
   ▲ you are here
```

---

## Table of Contents

1. [When to Use Brainstorm](#1-when-to-use-brainstorm)
2. [The 7-Step Workflow](#2-the-7-step-workflow)
3. [Three Entry Points](#3-three-entry-points)
4. [Design Doc Format](#4-design-doc-format)
5. [Self-Review Checklist](#5-self-review-checklist)
6. [Handoff to Genesis](#6-handoff-to-genesis)
7. [Design Principles](#7-design-principles)
8. [When to Skip Brainstorm](#8-when-to-skip-brainstorm)

---

## 1. When to Use Brainstorm

Use brainstorm when you arrive at AHVS with any of these:

| Starting point | Example | What brainstorm does |
|----------------|---------|---------------------|
| Data + vague idea | "I have `emails.csv` and want to classify intent" | Reads your data, proposes solver approaches, writes design |
| Existing repo to improve | "My RAG pipeline's precision is low" | Scans the repo, finds metrics/experiments, proposes strategies |
| Just an idea | "I want a toxicity detector but haven't started" | Asks clarifying questions, proposes approaches, writes design |

The key value: **you make informed decisions before spending compute.** Genesis, labeling, training, and AHVS cycles all cost time and money. Brainstorm costs only a conversation.

## 2. The 7-Step Workflow

| Step | What happens | Hard gate? |
|------|-------------|------------|
| 1. Understand what user has | Asks A/B/C: data, repo, or idea | -- |
| 2. Ask clarifying questions | One at a time, multiple-choice | -- |
| 3. Propose approaches | 2-3 options with trade-offs | -- |
| 4. Write design doc | Saved to `docs/ahvs/designs/` | -- |
| 5. Self-review | Checks for gaps, contradictions | -- |
| 6. User reviews design | Approve, revise, or restart | **Yes** |
| 7. Hand off to genesis | Provides pre-filled genesis command | **Yes** |

**Hard gate:** No genesis, no scaffolding, no solvers, no implementation until the user explicitly approves the design. This applies even for "simple" problems.

## 3. Three Entry Points

When brainstorm starts, it asks what you have:

```
What do you have so far?
  A) A data file (CSV, Parquet, etc.) and a problem idea
  B) An existing repo/project I want to improve with AHVS
  C) Just an idea — no data or code yet
```

### A) Data + idea

You provide the data file path. Brainstorm reads a sample (first 10-20 rows), checks schema, column names, row count, and class distribution. Then moves to clarifying questions.

### B) Existing repo

You provide the repo path. Brainstorm scans:
- Data files, eval scripts, existing models
- READMEs, notebooks, prior experiment notes
- Git history — recent work direction, abandoned approaches
- Domain — NLP classification, RAG, ML, etc.

### C) Just an idea

Brainstorm skips straight to clarifying questions to flesh out the concept. No scanning needed.

## 4. Design Doc Format

After the user selects an approach, brainstorm writes a design document to:

```
docs/ahvs/designs/YYYY-MM-DD-<topic>-design.md
```

The document contains:

| Section | What it covers |
|---------|---------------|
| **Problem** | What we're solving, in concrete terms |
| **Data** | What data exists, schema, size, quality assessment |
| **Approach** | Selected approach with rationale |
| **Target Metric** | Primary metric, target value, regression guards |
| **Constraints** | Budget, model restrictions, latency, scope boundaries |
| **Genesis Inputs** | Pre-filled values ready to copy into `/ahvs_genesis` |
| **Risks & Open Questions** | What could go wrong, what we don't know yet |
| **Prior Art** | What's been tried, lessons learned |

The **Genesis Inputs** section is the critical output — it directly feeds the next stage:

```markdown
## Genesis Inputs
- Problem description: Classify customer emails into intent categories
- Data path: /home/user/data/emails.csv
- Target metric: f1_weighted
- Execution mode: pipeline
- Classes: ["urgent", "question", "feedback", "spam"]
- Annotation model: gpt-4.1-mini (per cost policy)
```

## 5. Self-Review Checklist

Before presenting the design to the user, brainstorm checks:

| Category | What to look for |
|----------|-----------------|
| Completeness | TODOs, placeholders, "TBD", empty sections |
| Consistency | Contradicting requirements (e.g., "fast" + "use expensive model") |
| Clarity | Anything ambiguous enough to cause genesis to build the wrong thing |
| Scope | Is this one project or secretly two? |
| YAGNI | Over-engineered for the actual problem |

Issues are fixed before the user sees the design.

## 6. Handoff to Genesis

Once the user approves, brainstorm does **not** run genesis itself. Instead, it provides the exact command with all parameters pre-filled:

```
Your design is approved. To create the project, run:

  /ahvs_genesis

Or with the browser form:

  /ahvs_genesis:gui

Genesis inputs (from your design):
  Problem:     Classify customer emails into intent categories
  Data:        /home/user/data/emails.csv
  Metric:      f1_weighted
  Mode:        pipeline
  Classes:     ["urgent", "question", "feedback", "spam"]
```

The user decides when to run genesis. Brainstorm never auto-chains.

## 7. Design Principles

- **Break systems into smaller, well-bounded units** with single purposes
- **Each unit should be independently understandable and testable**
- **Use multiple-choice questions** when possible — reduces back-and-forth
- **Present 2-3 alternatives** before settling on one approach
- **Scale documentation to complexity** — simple problem = short doc
- **Follow existing codebase patterns** — don't reinvent what's already there
- **Avoid unrelated scope creep** — brainstorm is about THIS problem

## 8. When to Skip Brainstorm

Skip brainstorm and go directly to `/ahvs_genesis` when:

- You already know the problem, data, metric, and approach
- You've done this kind of project before and want to jump in
- You're re-creating a project with different parameters

Brainstorm is for when you need to **think it through first**.

---

## Usage

In Claude Code:
```
/ahvs_brainstorm
```

## LLM Cost Policy

When discussing approach costs, brainstorm references the project policy:
- Annotation/labeling: `gpt-4.1-mini` (never `gpt-4o`)
- AHVS orchestration: ACP (Claude Code subscription)
- Cost implications are included in approach trade-offs
