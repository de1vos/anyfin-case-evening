import sys
import pandas as pd

train = pd.read_csv("transactions.csv")   # paste "Copy CSV" into this file
test = pd.read_csv(sys.argv[1])           # python your-code.py validation.csv
# an API key, if you use one: os.environ["ANTHROPIC_API_KEY"]

def classify(row) -> str:
    text = row["text"].upper()
    if "SWISH" in text:
        return "transfer.person"
    return "unknown"                      # your classifier goes here

test["category"] = test.apply(classify, axis=1)
test[["id", "category"]].to_csv("predictions.csv", index=False)