"""GPT-2 backbones for all four tasks (Sections 4.1, 4.2, 5.1)."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer

PROMPT_TEMPLATE = (
    'Question 1: "{s1}"\nQuestion 2: "{s2}"\nAre these questions asking the same thing?'
)

# The report specifies the paraphrase prompt but not the sentiment one, so this
# mirrors its style. SST is 5-class, CFIMDB binary; both are single-sentence.
SENTIMENT_TEMPLATE = 'Review: "{s}"\nWhat is the sentiment of this review?'

NUM_LABELS = {"qqp": 2, "sst": 5, "cfimdb": 2}


def build_tokenizer(name: str = "gpt2") -> GPT2Tokenizer:
    tok = GPT2Tokenizer.from_pretrained(name)
    # GPT-2 ships without a pad token; reuse EOS and rely on the attention mask.
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


class GPT2Classifier(nn.Module):
    """Decoder-only GPT-2 with a linear head on the final non-pad token.

    y = W h_final + b,  W in R^{768 x num_labels}

    Shared by all three classification tasks; only num_labels changes
    (QQP 2, SST 5, CFIMDB 2).
    """

    def __init__(self, base: str = "gpt2", num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.base_name = base
        self.num_labels = num_labels
        self.gpt2 = GPT2Model.from_pretrained(base)
        self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        hidden = self.gpt2.config.n_embd  # 768 for gpt2-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)

        # Index of the last real (non-pad) token in each sequence.
        last_idx = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        h_final = hidden[batch_idx, last_idx]  # (B, H)

        return self.classifier(self.dropout(h_final))

    # ---- checkpoint I/O -------------------------------------------------
    def save(self, out_dir: str, half: bool = False) -> None:
        os.makedirs(out_dir, exist_ok=True)
        state = self.state_dict()
        if half:
            state = {k: v.half() for k, v in state.items()}
        torch.save(state, os.path.join(out_dir, "pytorch_model.bin"))
        with open(os.path.join(out_dir, "head_config.json"), "w") as f:
            json.dump(
                {"base": self.base_name, "num_labels": self.num_labels, "half": half},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, ckpt_dir: str, device: str = "cpu") -> GPT2ParaphraseClassifier:
        with open(os.path.join(ckpt_dir, "head_config.json")) as f:
            cfg = json.load(f)
        model = cls(base=cfg["base"], num_labels=cfg["num_labels"])
        state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        state = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        return model.to(device).eval()


def encode_pairs(tokenizer, s1_list, s2_list, max_length: int = 128):
    prompts = [PROMPT_TEMPLATE.format(s1=a, s2=b) for a, b in zip(s1_list, s2_list, strict=True)]
    return tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# Back-compat alias: the paraphrase task is the report's primary experiment.
GPT2ParaphraseClassifier = GPT2Classifier


def encode_single(tokenizer, texts, max_length: int = 128):
    """Encode single sentences for the sentiment tasks (SST, CFIMDB)."""
    prompts = [SENTIMENT_TEMPLATE.format(s=t) for t in texts]
    return tokenizer(
        prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


class GPT2SonnetLM(nn.Module):
    """GPT-2 with its language-modelling head, for Shakespearean sonnet generation.

    Unlike the classification tasks this keeps the tied output projection and
    trains on next-token prediction over the sonnet corpus.
    """

    def __init__(self, base: str = "gpt2"):
        super().__init__()
        self.base_name = base
        self.lm = GPT2LMHeadModel.from_pretrained(base)
        self.lm.config.pad_token_id = self.lm.config.eos_token_id

    def forward(self, input_ids, attention_mask, labels=None):
        return self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.9,
        top_p: float = 0.9,
        device: str = "cpu",
    ) -> str:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        out = self.lm.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.lm.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "sonnet_config.json"), "w") as f:
            json.dump({"base": self.base_name, "task": "sonnet"}, f, indent=2)

    @classmethod
    def load(cls, ckpt_dir: str, device: str = "cpu") -> GPT2SonnetLM:
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.base_name = "gpt2"
        model.lm = GPT2LMHeadModel.from_pretrained(ckpt_dir)
        return model.to(device).eval()
