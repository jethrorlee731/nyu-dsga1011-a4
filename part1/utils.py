import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import hashlib
import re
from functools import lru_cache
from nltk import pos_tag

random.seed(0)

# Experiment settings; these are starting values, not guaranteed thresholds.
SYNONYM_PROB = 0.00
TYPO_PROB = 0.45

# Nearby keys for vowels on a QWERTY keyboard.
_NEARBY_KEYS = {
    "a": "sq",
    "e": "wr",
    "i": "uo",
    "o": "ip",
    "u": "yi",
}

_PROTECTED_WORDS = {
    "no", "not", "never", "neither", "nor",
    "hardly", "barely", "without", "only",
}

# Skip inflected forms to avoid replacing "enjoyed" with "enjoy", etc.
_WORDNET_POS = {
    "NN": wordnet.NOUN,
    "VB": wordnet.VERB,
    "JJ": wordnet.ADJ,
    "RB": wordnet.ADV,
}

# Match HTML tags separately so they are preserved.
_TOKEN_PATTERN = re.compile(
    r"<[^>]*>|[A-Za-z]+(?:['’][A-Za-z]+)*"
)


@lru_cache(maxsize=20000)
def synonym_candidates(word, penn_tag):
    wn_pos = _WORDNET_POS.get(penn_tag)
    if wn_pos is None:
        return ()

    synsets = wordnet.synsets(word, pos=wn_pos)
    if not synsets:
        return ()

    # Restrict to one sense to reduce unrelated replacements.
    # This is still a heuristic, not contextual sense disambiguation.
    candidates = {
        lemma.name().lower()
        for lemma in synsets[0].lemmas()
        if lemma.name().isascii()
        and lemma.name().isalpha()
        and lemma.name().lower() != word
        and lemma.name().lower() not in _PROTECTED_WORDS
    }

    return tuple(sorted(candidates))

def _match_case(replacement, original):
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    result = dict(example)
    text = example["text"]

    # Reproducible per-review randomness, incorporating seed 38.
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big") ^ 38
    rng = random.Random(seed)

    # Exclude HTML tags from tagging and modification.
    matches = [
        match
        for match in _TOKEN_PATTERN.finditer(text)
        if not match.group(0).startswith("<")
    ]

    if not matches:
        return result

    words = [match.group(0) for match in matches]
    tagged_words = pos_tag(words)

    pieces = []
    previous_end = 0

    for match, (_, tag) in zip(matches, tagged_words):
        original = match.group(0)
        replacement = original

        eligible = (
            original.islower()
            and "'" not in original
            and "’" not in original
            and original not in _PROTECTED_WORDS
        )

        if eligible:
            # First, optionally replace with a single-word synonym.
            if rng.random() < SYNONYM_PROB:
                candidates = synonym_candidates(original, tag)
                if candidates:
                    replacement = rng.choice(candidates)

            # Then, optionally introduce one keyboard-neighbor typo.
            # Keep the first and last letters unchanged.
            if len(replacement) >= 5 and rng.random() < TYPO_PROB:
                positions = [
                    i
                    for i in range(1, len(replacement) - 1)
                    if replacement[i] in _NEARBY_KEYS
                ]

                if positions:
                    i = rng.choice(positions)
                    letters = list(replacement)
                    letters[i] = rng.choice(_NEARBY_KEYS[letters[i]])
                    replacement = "".join(letters)

        pieces.append(text[previous_end:match.start()])
        pieces.append(replacement)
        previous_end = match.end()

    pieces.append(text[previous_end:])
    result["text"] = "".join(pieces)

    text = result["text"]

    # Independently apply punctuation variation to 20% of reviews.
    # Modify only terminal punctuation, preserving trailing whitespace.
    if rng.random() < 0.20:
        ending = re.search(r"([.!])(\s*)$", text)
        if ending:
            replacement = "!!!" if ending.group(1) == "!" else ""
            text = (
                text[:ending.start()]
                + replacement
                + ending.group(2)
            )

    # Independently delete one article from 20% of reviews.
    # Skip HTML tags so their contents remain unchanged.
    if rng.random() < 0.20:
        candidates = [
            match
            for match in re.finditer(
                r"<[^>]*>|\b(?:a|an|the)\b",
                text,
                flags=re.IGNORECASE,
            )
            if not match.group(0).startswith("<")
        ]

        if candidates:
            chosen = rng.choice(candidates)
            start, end = chosen.span()

            # Remove an adjacent space without joining neighboring words.
            if end < len(text) and text[end] == " ":
                end += 1
            elif start > 0 and text[start - 1] == " ":
                start -= 1

            text = text[:start] + text[end:]

    result["text"] = text

    # Preserve the original label.
    return result
