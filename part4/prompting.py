"""
Prompting utilities for multiple-choice QA.
Example submission.
"""
import torch
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax


class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "instruction": "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: {question}\n\n{choices_formatted}\n\nSelect the answer:",
        "simple": "{context}\n{question}\n{choices_formatted}\nThe answer is",
        # Leaving "micah" alone; but note it hardcodes an example with "Answer: 0"
        "micah": (
            "Context: On May 21, 2013, NFL owners at their spring meetings in Boston voted and awarded the game to Levi's Stadium. "
            "The $1.2 billion stadium opened in 2014. It is the first Super Bowl held in the San Francisco Bay Area since Super Bowl XIX in 1985, "
            "and the first in California since Super Bowl XXXVII took place in San Diego in 2003.\n"
            "Question: Where did the spring meetings of the NFL owners take place?\n"
            "Options:\n"
            "0. Boston\n1. Super Bowl XXXVII\n2. cholera\n3. between June and September\n"
            "Answer: 0\n\n"
            "Context: {context}\nQuestion: {question}\nOptions:\n{choices_formatted}\nAnswer:"
        ),
    }

    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "index"):
        # choice_format: "index" (0,1,2,3,...) or "letter" (A,B,C,...)
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format

    def _format_choices(self, choices: List[str]) -> str:
        if self.choice_format == "letter":
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        else:
            labels = [str(i) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

    def format(self, context: str, question: str, choices: List[str], **kwargs) -> str:
        return self.template.format(
            context=context,
            question=question,
            choices_formatted=self._format_choices(choices),
            **kwargs,
        )

    def format_with_answer(self, context: str, question: str, choices: List[str], answer_idx: int) -> str:
        prompt = self.format(context, question, choices)
        if self.choice_format == "letter":
            label = chr(ord("A") + answer_idx)
        else:
            label = str(answer_idx)
        return f"{prompt} {label}"


class PromptingPipeline:
    def __init__(self, model, tokenizer, template: Optional[PromptTemplate] = None, device: str = "cuda"):
        self.model = model.to(device) if hasattr(model, "to") else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic", choice_format="index")
        self.device = device
        self._setup_choice_tokens()

    def _setup_choice_tokens(self, max_choices: int = 8):
        """
        Precompute token ids for choice labels:
          index mode: "0","1","2",...
          letter mode: "A","B","C",...
        We try both "" and " " prefix.
        """
        self.choice_tokens = {}

        # build label list based on template mode
        if getattr(self.template, "choice_format", "index") == "letter":
            labels = [chr(ord("A") + i) for i in range(max_choices)]
        else:
            labels = [str(i) for i in range(max_choices)]

        for label in labels:
            for prefix in ["", " "]:
                token_ids = self.tokenizer.encode(prefix + label)
                if token_ids:
                    # NOTE: still assumes single-token label is best-effort.
                    # If your tokenizer splits these, consider scoring full continuation.
                    self.choice_tokens[label] = token_ids[-1]
                    break

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], return_probs: bool = False):
        self.model.eval()
        prompt = self.template.format(context, question, choices)
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)

        logits = self.model(input_ids)  # expected [1, T, V]
        logits = logits[:, -1, :]       # [1, V]

        # build labels corresponding to choices
        if getattr(self.template, "choice_format", "index") == "letter":
            choice_labels = [chr(ord("A") + i) for i in range(len(choices))]
        else:
            choice_labels = [str(i) for i in range(len(choices))]

        choice_logits = []
        vocab_size = logits.size(-1)

        for label in choice_labels:
            tok = self.choice_tokens.get(label, None)
            if tok is None or not (0 <= int(tok) < vocab_size):
                choice_logits.append(float("-inf"))
            else:
                choice_logits.append(logits[0, int(tok)].item())

        choice_logits = torch.tensor(choice_logits, device="cpu")
        probs = softmax(choice_logits, dim=-1)
        prediction = int(probs.argmax().item())

        if return_probs:
            return prediction, probs.tolist()
        return prediction

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8) -> List[int]:
        # minimal edit: still simple loop
        return [self.predict_single(ex["context"], ex["question"], ex["choices"]) for ex in examples]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
    predictions = pipeline.predict_batch(examples, batch_size)
    correct = sum(
        1
        for p, ex in zip(predictions, examples)
        if ex.get("answer", -1) >= 0 and p == ex["answer"]
    )
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}