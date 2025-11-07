# Text Summarizer (standalone)

A small, dependency-free extractive text summarizer implemented in `summarizer.py`.

Features
- Extractive summarization: selects informative sentences based on normalized word frequency.
- Small, self-contained: no external Python packages required.
- Options: ratio of sentences to keep, soft word cap, deduplication toggle and threshold, JSON output.

Usage (PowerShell)

```powershell
python e:\Text_Summarizer\summarizer.py --input "e:\Text_Summarizer\article.txt" --ratio 0.3 --json
```

Example input (`article.txt`)

```
Renewable energy has become an essential part of the global response to climate change. Over the last decade, wind and solar power have seen rapid cost declines and deployment increases, enabling many countries to reduce their dependence on fossil fuels. Grid operators are adapting, investing in storage and demand-response technologies to manage variability.

However, integration challenges remain. Long-term planning must account for seasonal storage and transmission upgrades. Policymakers are also examining market reforms to ensure adequate incentives for flexible capacity and investment in resilient infrastructure. Community engagement and workforce development are critical to achieve a just transition.

Innovation continues at both utility and distributed scales. Advances in battery chemistry, grid-scale thermal storage, and green hydrogen pilot projects promise to expand options for decarbonization. Meanwhile, efficiency improvements and electrification of transport and heating further reduce overall demand growth.

In summary, renewable energy is not a single solution but a portfolio of approaches that must be coordinated through policy, markets, and engineering to meet climate goals while maintaining reliability and affordability.
```

Example output (JSON)

```json
{
  "summary": "Over the last decade, wind and solar power have seen rapid cost declines and deployment increases, enabling many countries to reduce their dependence on fossil fuels. Policymakers are also examining market reforms to ensure adequate incentives for flexible capacity and investment in resilient infrastructure. Advances in battery chemistry, grid-scale thermal storage, and green hydrogen pilot projects promise to expand options for decarbonization. In summary, renewable energy is not a single solution but a portfolio of approaches that must be coordinated through policy, markets, and engineering to meet climate goals while maintaining reliability and affordability.",
  "sentences_selected": 4,
  "total_sentences": 11,
  "input_words": 169,
  "output_words": 94,
  "reduction_ratio": 0.4438,
  "algorithm": "frequency_v2"
}
```

Notes & next steps
- You can tune `--ratio`, `--max_words`, and `--dedupe-threshold` to change summarization behaviour.
- For production-quality tokenization and sentence-splitting, consider adding an optional dependency such as spaCy or NLTK and wiring it behind a CLI flag.

License: MIT (use freely)
