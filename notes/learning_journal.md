# Learning Journal

Use this file as a chronological log. Keep each entry short and actionable.

## Entry Template

Date:
Topic:
Goal:
What I learned:
What is still unclear:
Code touched:
Experiment config:
Observed result:
Mistake and fix:
Next action:

---

## 2026-03-23

Date: 2026-03-23
Topic: Build a structured note system for MeteorMind
Goal: Prepare a reusable note workflow for Pretrain -> SFT -> PPO/DPO/GRPO learning
What I learned:
- The project can benefit from phase-based notes instead of scattered text.
- A journal + stage summary structure is easier to maintain.
What is still unclear:
- Exact training/evaluation metrics to standardize for each stage.
Code touched:
- Added `notes/` folder and initial markdown files.
Experiment config:
- N/A
Observed result:
- Notes now have a fixed structure and can scale with progress.
Mistake and fix:
- None yet.
Next action:
- Start filling `stage_pretrain.md` with tokenizer, objective, and data pipeline notes.

---

## 2026-03-30

Date: 2026-03-30
Topic: Pretrain project documentation consolidation
Goal: Consolidate repository description and extract implementation Tips into structured notes
What I learned:
- Current pretrain pipeline is runnable end-to-end with tokenizer, dataset, training loop, and checkpoint resume.
- Code comments/Tips are valuable prompts for first-principles understanding and should be tracked separately.
What is still unclear:
- Some model details (RoPE scaling, GLU variant, Pre-Norm rationale) still need deeper math derivations.
Code touched:
- Updated `README.md`
- Updated `notes/stage_pretrain.md`
- Updated `notes/README.md`
- Added `notes/pretrain_tips_digest.md`
Experiment config:
- N/A (documentation and notes pass)
Observed result:
- Project now has a clearer onboarding README and a dedicated Tips digest for pretrain code.
Mistake and fix:
- Previous docs did not map Tips to actionable follow-ups; fixed by adding question -> explanation -> next-action format.
Next action:
- Start SFT training script draft and keep the same note discipline.

---

## 2026-03-30 (Update)

Date: 2026-03-30
Topic: Pretrain tips deepening
Goal: Upgrade Tips notes from summary style to formula-and-shape learning style
What I learned:
- Tips become much more actionable when each item includes equation + tensor shape + minimal example.
- RoPE/Attention related confusion can be reduced quickly with a fixed "shape trace" template.
What is still unclear:
- Need one short benchmark run to verify alignment-friendly dimensions (intermediate size alignment) on real hardware.
Code touched:
- Updated `notes/pretrain_tips_digest.md`
- Updated `notes/learning_journal.md`
Experiment config:
- N/A (note refinement pass)
Observed result:
- Pretrain Tips file is now usable as a focused study checklist for code reading and future annotation.
Mistake and fix:
- Prior version lacked explicit formulas; fixed by adding math and sanity checks for each Tip.
Next action:
- Start drafting SFT trainer skeleton and add corresponding stage note links.
