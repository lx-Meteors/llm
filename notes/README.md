# MeteorMind Learning Notes

This folder stores a structured learning record for the full LLM training pipeline:

- Main handbook: `MeteorMind_全流程学习手册.md`
- Pretrain
- SFT
- Alignment and preference optimization (PPO, DPO, GRPO)
- Evaluation and iteration

## How to use

1. Write one short entry after each study session in `learning_journal.md`.
2. Move stable conclusions into stage files:
   - `stage_pretrain.md`
   - `stage_sft.md`
   - `stage_alignment_rl.md`
3. Read the end-to-end handbook first if you are new:
   - `MeteorMind_全流程学习手册.md`
4. For code-level questions and implementation Tips, maintain:
   - `pretrain_tips_digest.md`
5. Keep each note with:
   - Concept
   - Why it matters
   - Implementation in this repo
   - Common pitfalls
   - Next experiment

## Note quality checklist

- Explain from first principles.
- Include formulas only when needed.
- Add one practical code or training link to this repo.
- Add one reproducible experiment setting.
- Add one failure case and fix.

## Recommended workflow

1. Train or read code.
2. Add one short `learning_journal.md` entry.
3. Promote stable conclusions to stage notes.
4. If confusion appears in code comments/Tips, record it in `pretrain_tips_digest.md` first, then backfill with equations/examples.
