# Decisions

Architecture and process decisions. Newest at the top. Format: date, decision, rationale.

## 2026-07-03

- **Fail-loud real engines:** `EngineConfig.allow_fallback` defaults False; silent baseline/mock under another engine name is forbidden for publishable runs. Rationale: mislabeled rows invalidate GPU comparisons.
- **Primary engines baseline+vllm:** HF proxies use honest names (`hf_prefix_cache`, `hf_tool_interrupt`); `preble`/`infercept` remain aliases only. Rationale: avoid claiming paper systems that are not integrated.
- **Game-level agent metrics:** `rounds` is games played; RPS+ uses 20 turns per game. Rationale: turn wins were mislabeled as games.
- **Project agent toolkit:** Context via `AGENTS.md` + `.cursor/rules/`, persistent memory in `.cursor/memory/`, workflows in `.cursor/skills/`. Rationale: superpowers-style continuity across sessions without changing runtime code.

## 2026-06-24

- **Centralize `BEST_COUNTER`:** Single source in `environments/rps_plus.py`; agents and impostor code import it (or must match POWER->POWER). Rationale: duplicated POWER->RECHARGE maps caused guaranteed losses and wrong bias metrics.
- **Mock-first CI:** All engines and benchmarks must run with `--model mock` on CPU. Real GPU engines fall back in `setup_inference_engines`. Rationale: contributors and CI have no GPU; publishable numbers come from Brev/GPU runs separately.
