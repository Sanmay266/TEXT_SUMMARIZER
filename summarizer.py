#!/usr/bin/env python3
"""
summarize.py — Simple NLP Text Summarizer (no external libraries)

Extractive summarization that picks the most informative sentences based on
normalized word frequency, with a light preference for sentences near the
average length and optional near-duplicate filtering.

Usage
-----
Summarize a file (keep ~20% of sentences):
    python summarize.py --input article.txt --ratio 0.2

Summarize raw text and cap to ~120 words:
    python summarize.py --text "Very long text ..." --max_words 120

Return JSON (for automation):
    python summarize.py --input article.txt --json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import statistics
import string
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# Expanded (still small) English stopword list — extend as needed
_STOPWORDS = set(map(str.lower, '''
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into
through during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don dont should shouldn't
now ll ve re m d s y aren't can't couldn't won't wouldn't i'm you're he's she's it's we're
they've i've we've wasn't weren't isn't aren't
'''.split()))

# Improved sentence splitter: split on end punctuation followed by whitespace
# and a likely sentence start (capital/digit/quote/paren) to reduce splitting on
# abbreviations and initials. This is still heuristic but better than the very
# naive version.
_SENT_SPLIT_REGEX = re.compile(r'(?<=[\.\?!…])\s+(?=[A-Z0-9"\'\(\[\u2018\u201C])')

# Word tokenizer: allow ASCII letters, common Latin accents, digits, hyphens and apostrophes
_WORD_REGEX = re.compile(r"[A-Za-z0-9\u00C0-\u017F'-]+")


@dataclass
class SummaryResult:
    summary: str
    sentences_selected: int
    total_sentences: int
    input_words: int
    output_words: int
    reduction_ratio: float  # 0.78 means 78% shorter
    algorithm: str = "frequency_v2"


def _sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_REGEX.split(text)
    sentences = [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]
    return sentences


def _word_tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_REGEX.findall(text)]


def _build_freqs(tokens: Iterable[str]) -> Dict[str, float]:
    freqs: Dict[str, int] = {}
    for tok in tokens:
        if not tok:
            continue
        if tok in _STOPWORDS:
            continue
        if all(ch in string.punctuation for ch in tok):
            continue
        freqs[tok] = freqs.get(tok, 0) + 1

    if not freqs:
        return {}

    # Normalize by max frequency and dampen with log
    max_f = max(freqs.values())
    norm = {w: (math.log(1 + f) / math.log(1 + max_f)) for w, f in freqs.items()}

    # Median-centered robust scaling to reduce outliers
    values = list(norm.values())
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values]) or 1.0
    scaled = {w: 0.5 + 0.5 * ((v - median) / (6 * mad)) for w, v in norm.items()}
    return {w: max(0.0, min(1.0, s)) for w, s in scaled.items()}


def _score_sentence(sent: str, word_freqs: Dict[str, float], avg_len: float) -> float:
    tokens = _word_tokenize(sent)
    if not tokens:
        return 0.0
    score = sum(word_freqs.get(t, 0.0) for t in tokens)
    # Prefer sentences near the average length
    length = len(tokens)
    if avg_len > 0:
        penalty = math.exp(-((length - avg_len) ** 2) / (2 * (0.6 * avg_len + 1) ** 2))
        score *= (0.6 + 0.4 * penalty)
    return score


def summarize_text(
    text: str,
    ratio: float = 0.2,
    max_words: Optional[int] = 120,
    min_sentences: int = 2,
    max_sentences: Optional[int] = None,
    dedupe: bool = True,
    dedupe_threshold: float = 0.75,
) -> SummaryResult:
    """Return an extractive summary of `text`."""
    sentences = _sentence_split(text)
    total_sentences = len(sentences)
    if total_sentences == 0:
        return SummaryResult("", 0, 0, 0, 0, 0.0)

    all_tokens = _word_tokenize(text)
    word_freqs = _build_freqs(all_tokens)
    if not word_freqs:
        sel = sentences[:max(min_sentences, 1)]
        out = " ".join(sel)
        iw = len(all_tokens)
        ow = len(_word_tokenize(out))
        reduction = 1.0 - (ow / max(1, iw))
        return SummaryResult(out, len(sel), total_sentences, iw, ow, reduction)

    avg_len = statistics.mean([len(_word_tokenize(s)) for s in sentences]) if sentences else 0.0

    scored = []
    for idx, s in enumerate(sentences):
        sc = _score_sentence(s, word_freqs, avg_len)
        scored.append((sc, idx, s))

    # how many sentences to keep
    k = max(min_sentences, max(1, int(math.ceil(total_sentences * ratio))))
    if max_sentences is not None:
        k = min(k, max_sentences)
    k = min(k, total_sentences)

    topk = heapq.nlargest(k, scored, key=lambda x: (x[0], -x[1]))

    # Dedupe near-duplicates by Jaccard overlap (ignores stopwords where possible)
    if dedupe and len(topk) > 1:
        selected = []
        used_sets = []
        for sc, idx, s in sorted(topk, key=lambda t: (-t[0], t[1])):
            toks = set(t for t in _word_tokenize(s) if t not in _STOPWORDS)
            # fallback to full tokens if removing stopwords yields empty set
            if not toks:
                toks = set(_word_tokenize(s))
            if toks:
                too_similar = any(len(toks & us) / max(1, len(toks | us)) > dedupe_threshold for us in used_sets)
                if too_similar:
                    continue
                used_sets.append(toks)
            selected.append((idx, s))
            if len(selected) == k:
                break
    else:
        selected = [(idx, s) for _, idx, s in topk]

    selected.sort(key=lambda x: x[0])  # keep original order
    ordered_sents = [s for _, s in selected]

    # Optional soft word cap (stop at a sentence boundary)
    if max_words is not None and max_words > 0:
        capped = []
        total = 0
        for s in ordered_sents:
            tok_count = len(_word_tokenize(s))
            if total + tok_count <= max_words or len(capped) < 2:
                capped.append(s)
                total += tok_count
            else:
                break
        ordered_sents = capped

    summary = " ".join(ordered_sents).strip()
    input_words = len(all_tokens)
    output_words = len(_word_tokenize(summary))
    reduction = 1.0 - (output_words / max(1, input_words))

    return SummaryResult(
        summary=summary,
        sentences_selected=len(ordered_sents),
        total_sentences=total_sentences,
        input_words=input_words,
        output_words=output_words,
        reduction_ratio=reduction,
    )


def main():
    parser = argparse.ArgumentParser(description="Extractive text summarizer (no external deps).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a text file to summarize.")
    group.add_argument("--text", type=str, help="Raw text to summarize (quote the string).")
    parser.add_argument("--ratio", type=float, default=0.2, help="Fraction of sentences to keep (0-1]. Default 0.2.")
    parser.add_argument("--max_words", type=int, default=120, help="Cap summary to ~N words (best-effort).")
    parser.add_argument("--min_sentences", type=int, default=2, help="Minimum sentences in summary.")
    parser.add_argument("--max_sentences", type=int, default=None, help="Maximum sentences in summary.")
    parser.add_argument("--json", action="store_true", help="Return JSON instead of plain text.")
    parser.add_argument("--no-dedupe", action="store_false", dest="dedupe", help="Disable near-duplicate filtering.")
    parser.add_argument("--dedupe-threshold", type=float, default=0.75, help="Jaccard threshold for dedupe (0-1). Default 0.75.")
    args = parser.parse_args()

    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file '{args.input}': {e}", file=sys.stderr)
            sys.exit(2)
    else:
        text = args.text or ""

    res = summarize_text(
        text=text,
        ratio=max(1e-6, min(1.0, args.ratio)),
        max_words=args.max_words,
        min_sentences=max(1, args.min_sentences),
        max_sentences=args.max_sentences,
        dedupe=args.dedupe,
        dedupe_threshold=min(max(0.0, args.dedupe_threshold), 1.0),
    )

    if args.json:
        print(json.dumps({
            "summary": res.summary,
            "sentences_selected": res.sentences_selected,
            "total_sentences": res.total_sentences,
            "input_words": res.input_words,
            "output_words": res.output_words,
            "reduction_ratio": round(res.reduction_ratio, 4),
            "algorithm": res.algorithm
        }, ensure_ascii=False, indent=2))
    else:
        print(res.summary)
        print("\n---")
        print(f"Sentences: {res.sentences_selected}/{res.total_sentences} | "
              f"Words: {res.output_words}/{res.input_words} | "
              f"Reduction: {res.reduction_ratio:.0%}")


if __name__ == "__main__":
    main()