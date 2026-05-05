"""Wrappers around a HuggingFace causal LM with optional LoRA adapters.

GPU-only at training time; CPU inference works for tiny test models.
Imports of heavy libraries are lazy so the rest of the project can be used
without HuggingFace installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import Agent
from agents.checkpoints import resolve_checkpoint
from environments.rps_plus import Move, N_MOVES


@dataclass
class SFTConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    quantize_4bit: bool = False


def load_model_and_tokenizer(cfg: SFTConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {}
    if cfg.quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    return model, tokenizer


def attach_lora(model, cfg: SFTConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg.quantize_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    return get_peft_model(model, lora)


class SFTAgent(Agent):
    """Inference-time agent that wraps a fine-tuned LM."""

    name = "sft"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        history_window: int = 5,
        max_new_tokens: int = 4,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        self.model = None
        self.tokenizer = None
        self.rng = np.random.default_rng(seed)
        self.history_window = history_window
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._history: list[tuple[Move, Move, int]] = []
        self.checkpoint_path = resolve_checkpoint("sft") if checkpoint is None else Path(checkpoint)

        if self.checkpoint_path is not None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(str(self.checkpoint_path))
                self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.model.eval()
            except Exception:
                self.model = None
                self.tokenizer = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._history.clear()

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        if agent_move not in range(N_MOVES) or opponent_move not in range(N_MOVES):
            return
        self._history.append((Move(agent_move), Move(opponent_move), outcome))

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        if self.model is not None and self.tokenizer is not None:
            from agents.sft.data import format_prompt
            import torch

            agent_energy = int(round(obs[0] * 5))
            opponent_energy = int(round(obs[1] * 5))
            prompt = format_prompt(agent_energy, opponent_energy, self._history, self.history_window)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            return parse_move(text, info)
        
        # Heuristic fallback
        from collections import Counter
        legal = set(info.get("legal_moves") or list(range(N_MOVES)))
        if not self._history:
            return int(self.rng.choice(list(legal)))
        
        # Approximate SFT behavior by countering the opponent's most frequent move
        from environments.rps_plus import BEST_COUNTER
        opp_moves = [h[1] for h in self._history]
        mode = Counter(opp_moves).most_common(1)[0][0]
        counter = BEST_COUNTER[mode]
        return int(counter) if int(counter) in legal else int(self.rng.choice(list(legal)))


def parse_move(text: str, info: dict[str, Any]) -> int:
    legal = info.get("legal_moves") or list(range(N_MOVES))
    text_upper = text.strip().upper()
    for m in Move:
        if m.name in text_upper and int(m) in legal:
            return int(m)
    return int(legal[0])
