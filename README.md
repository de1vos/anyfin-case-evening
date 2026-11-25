# anyfin-case-evening
Anyfin case evening with synthetic data

Hybrid approach:

Step 1: Clean the text: Remove dates and numbers. so Kortköp 260501 WILLYS SOLNA becomes WILLYS SOLNA. Also strip payment-provider prefixes (KLARNA*, ZETTLE_*, SUMUP *, IZ *, PAYPAL *.

First: Rules-based first
Create Rules-lexicon
Take the rule "text contains ICA → spend.groceries":
1. Find all training rows containing ICA. Say there are 20.
2. Compare the rule's label to the real label on each. Say 19 are groceries and 1 is something else.
3. The rule is right 19 out of 20 times (95%), so I keep it.
If a rule were right only 60% of the time, it isn't obvious. I'd remove it and let the model handle those rows.

Second: Small-model classifier  with linear regression (character n-gram logistic regression)
1. Turn the text into counts of short letter chunks (for example WIL, ILL, LLY).
2. Learn a weight for each chunk per category.
3. For new text, add up the weights. This gives a probability for each category.

Answer unknown when unsure

If the model's confidence is under about 60%, output unknown.

Test on unseen merchants

Hide all rows of some merchants, train on the rest, and score on the hidden ones. Repeat with different merchants and average. This imitates the real test set, where the shops are mostly new. A random split would give an inflated score, because the model would have seen the same shops in training.

Cost at 10 million rows a month: about 1 ms per row on one CPU core, so roughly 3 CPU-hours (under $1) a month. The code makes no API calls, so a run on their 5,517 hidden rows takes seconds and costs nothing.
With another day: a bigger merchant lexicon, a better-calibrated model, and a cheap LLM call only for the rows that are still unknown.
