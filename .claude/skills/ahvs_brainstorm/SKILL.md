---
name: ahvs_brainstorm
description: >-
  Pre-genesis brainstorming — explores the user's problem space, data, and
  goals before committing to a solver or project structure. Proposes 2-3
  approaches with trade-offs, writes a design doc, and hands off to genesis
  only after the user approves. Triggers on "brainstorm", "ahvs brainstorm",
  "help me figure out what to build", "explore my data", "what approach
  should I take", "design before building", or any pre-genesis exploration.
---

# AHVS Brainstorm

A structured design-first workflow that answers **"what should we build and
why?"** before genesis answers **"how do we scaffold it?"**

```
brainstorm → genesis → onboarding → ahvs cycles
```

## Hard Gate

**Do NOT invoke genesis, write any project scaffolding, run any solver, or
take any implementation action until you have presented a design and the user
has explicitly approved it.**

This applies even for "simple" problems. Unexamined assumptions cause wasted
compute and dead-end projects.

## What This Skill Does and Does Not Do

**Does:**
1. Explore project context — data files, existing code, docs, domain
2. Ask clarifying questions (one at a time, multiple-choice when possible)
3. Propose 2-3 approaches with trade-offs and a recommendation
4. Write a design doc to `docs/ahvs/designs/YYYY-MM-DD-<topic>-design.md`
5. Self-review the design for gaps, contradictions, and missing info
6. Wait for user approval before transitioning to genesis

**Does not:**
1. Run any solver, train any model, or label any data
2. Create `.ahvs/` directories or baseline files — that's genesis/onboarding
3. Auto-chain into genesis — the user decides when to proceed
4. Generate output paths — the user provides those during genesis

## Workflow

### Step 1: Explore Context

Understand what the user has before proposing anything. Inspect:

- **Data files** — read samples, check schema, row counts, class distribution
- **Existing code** — any models, pipelines, eval scripts already present?
- **Documentation** — READMEs, notebooks, prior experiment notes
- **Git history** — recent work direction, abandoned approaches
- **Domain** — what kind of problem is this? (NLP classification, RAG, ML, etc.)

Use Glob, Read, and Grep to scan. Don't ask the user for info you can discover.

### Step 2: Ask Clarifying Questions

Ask **one question at a time**. Prefer multiple-choice format:

```
What's the primary goal?
  A) Classify text into categories (e.g., intent detection)
  B) Extract structured info from unstructured text
  C) Improve an existing RAG/retrieval pipeline
  D) Something else — describe it
```

Key questions to resolve (skip any already answered by context):

1. **What problem are you solving?** — in the user's own words
2. **What does success look like?** — target metric, business outcome
3. **What constraints exist?** — cost budget, model restrictions, latency, privacy
4. **What's been tried before?** — prior approaches, what worked/failed
5. **Who consumes the output?** — downstream system, human reviewers, API

### Step 3: Propose Approaches

Present **2-3 concrete approaches** with:

| Approach | Description | Pros | Cons | Recommended? |
|----------|-------------|------|------|--------------|
| A | ... | ... | ... | ... |
| B | ... | ... | ... | ... |
| C | ... | ... | ... | ... |

For each approach, address:
- **Solver/method** — KD classifier, fine-tuning, rules-based, RAG, etc.
- **Data requirements** — what labeling/annotation is needed?
- **Cost estimate** — rough LLM cost for labeling, training time
- **Expected metric range** — based on data quality and problem difficulty
- **AHVS-ability** — how well does this approach support iterative optimization?

End with a clear recommendation and why.

### Step 4: Write Design Doc

After the user selects an approach, write a design document:

**Path:** `docs/ahvs/designs/YYYY-MM-DD-<topic>-design.md`

**Structure:**

```markdown
# <Topic> — AHVS Design

## Problem
<What we're solving, in concrete terms>

## Data
<What data exists, schema, size, quality assessment>

## Approach
<Selected approach with rationale>

## Target Metric
<Primary metric, target value, regression guards>

## Constraints
<Budget, model restrictions, latency, scope boundaries>

## Genesis Inputs
<Pre-filled values for the genesis step>
- Problem description: ...
- Data path: ...
- Target metric: ...
- Execution mode: pipeline | agent
- Classes: [...] or auto-detect
- Annotation model: gpt-4.1-mini (per cost policy)

## Risks & Open Questions
<What could go wrong, what we don't know yet>

## Prior Art
<What's been tried, lessons learned>
```

### Step 5: Self-Review

Before showing the design to the user, check for:

| Category | What to Look For |
|----------|------------------|
| Completeness | TODOs, placeholders, "TBD", empty sections |
| Consistency | Contradicting requirements (e.g., "fast" + "use expensive model") |
| Clarity | Anything ambiguous enough to cause genesis to build the wrong thing |
| Scope | Is this one project or secretly two? |
| YAGNI | Over-engineered for the actual problem |

Fix any issues before presenting.

### Step 6: User Reviews Design

Present the design doc and explicitly ask:

```
Design written to docs/ahvs/designs/YYYY-MM-DD-<topic>-design.md

Please review. When you're ready:
  - "approved" — I'll tell you the genesis command to run next
  - "change X" — I'll revise the design
  - "start over" — I'll propose new approaches
```

### Step 7: Hand Off to Genesis

Once approved, **do not run genesis yourself**. Instead, provide the exact
command with all parameters pre-filled from the design:

```
Your design is approved. To create the project, run:

  /ahvs_genesis

Or with the browser form:

  /ahvs_genesis:gui

Genesis inputs (from your design):
  Problem:     <from design>
  Data:        <from design>
  Metric:      <from design>
  Mode:        <from design>
  Classes:     <from design>
```

## Design Principles

- **Break systems into smaller, well-bounded units** with single purposes
- **Each unit should be independently understandable and testable**
- **Use multiple-choice questions** when possible — reduces back-and-forth
- **Present 2-3 alternatives** before settling on one approach
- **Scale documentation to complexity** — simple problem = short doc
- **Follow existing codebase patterns** — don't reinvent what's already there
- **Avoid unrelated scope creep** — brainstorm is about THIS problem

## LLM Cost Policy

When discussing approach costs, reference the project policy:
- Annotation/labeling: `gpt-4.1-mini` (never `gpt-4o`)
- AHVS orchestration: ACP (Claude Code subscription)
- Include cost implications in approach trade-offs
