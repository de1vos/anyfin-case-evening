"""Evaluation on the training set, holding out whole merchants (GroupKFold)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import classify as C

df = pd.read_csv("transactions.csv")
df["group"] = df["text"].map(C.core)

# --- rules alone, on all training rows (optimistic: I wrote them after reading this data)
d2 = C.add_counterparts(df)
rl = [C.rule_label(r) for _, r in d2.iterrows()]
fired = np.array([x is not None for x in rl])
correct = np.array([x == y for x, y in zip(rl, df.category)])
print(f"RULES   coverage {fired.mean():.1%}   precision when fired {correct[fired].mean():.1%}")
bad = d2[fired & ~correct].assign(pred=[x for x, f, c in zip(rl, fired, correct) if f and not c])
print(bad[["text", "category", "pred"]].to_string() if len(bad) else "  (no rule errors)")

# --- structural rules only (no merchant lexicon): shows what the lexicon adds
C.USE_LEXICON = False
rl0 = [C.rule_label(r) for _, r in d2.iterrows()]
f0 = np.array([x is not None for x in rl0]); c0 = np.array([x == y for x, y in zip(rl0, df.category)])
print(f"STRUCTURAL RULES ONLY  coverage {f0.mean():.1%}  precision {c0[f0].mean():.1%}")
C.USE_LEXICON = True

# --- model alone, merchants held out (honest)
gkf = GroupKFold(n_splits=5)
prob = np.zeros((len(df), df.category.nunique()))
classes = sorted(df.category.unique())
for tr, te in gkf.split(df, df.category, df.group):
    m = C.fit(df.iloc[tr])
    p = m.predict_proba(C.featurize(df.iloc[te]))
    for j, c in enumerate(m.classes_):
        prob[te, classes.index(c)] = p[:, j]
pred = np.array(classes)[prob.argmax(1)]
conf = prob.max(1)
print("\nMODEL ONLY (unseen merchants), by confidence threshold")
for th in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
    keep = conf >= th
    print(f"  th={th:.1f}  coverage {keep.mean():.1%}  precision {np.mean(pred[keep] == df.category[keep]):.1%}"
          f"  score(correct=+1, wrong=-1) {np.sum(np.where(keep, np.where(pred == df.category, 1, -1), 0)) / len(df):+.3f}")

# --- full pipeline, merchants held out for the model part (rules fixed)
print("\nFULL PIPELINE (rules + model + unknown), model trained on unseen merchants")
for th in [0.4, 0.5, 0.6, 0.7]:
    out = []
    for tr, te in gkf.split(df, df.category, df.group):
        m = C.fit(df.iloc[tr])
        labs, src = C.predict(df.iloc[te], m, th)
        out.append(pd.DataFrame({"i": te, "lab": labs, "src": src}))
    o = pd.concat(out).sort_values("i")
    o["y"] = df.category.values[o.i]
    ans = o.lab != "unknown"
    print(f"  th={th:.1f}  answered {ans.mean():.1%}  precision {np.mean(o.lab[ans] == o.y[ans]):.1%}"
          f"  accuracy(unknown=wrong) {np.mean(o.lab == o.y):.1%}")
    if th == 0.5:
        print("   by source:", o.groupby("src").apply(lambda g: f"n={len(g)} acc={np.mean(g.lab == g.y):.1%}").to_dict())
        print("   wrong answers:")
        w = o[ans & (o.lab != o.y)]
        print(pd.DataFrame({"text": df.text.values[w.i], "true": w.y.values, "pred": w.lab.values, "src": w.src.values}).to_string())
