**Run:** `pip install -r requirements.txt && python classify.py validation-set.csv` (writes `predictions.csv` as `id,category`; trains on `transactions.csv` in ~1 s).
**Approach:** clean the text, then rules for structure (savings and own-account pairing, Swish names/messages, card bills, salary, cash), then a merchant keyword lexicon, then a small char-n-gram logistic regression, then `unknown` below 70% confidence. No API, no key.
**Cost at 10M rows/month:** about 1 ms/row on one CPU core, so roughly 3 CPU-hours (under $1) a month; the validation set runs in about 2 s.
**With another day:** a bigger lexicon from real merchant data, a calibrated model (it is overconfident on unseen names), and a cheap LLM call only for the 10–15% still `unknown`.
**Caveat:** the lexicon was written after reading the training merchants, so the training score (about 100%) is optimistic; the model alone on unseen merchants is about 60% precise.
