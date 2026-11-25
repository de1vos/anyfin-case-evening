"""Transaction classifier: rules -> small model -> 'unknown'.

Usage:  python classify.py test.csv [train.csv]   ->  writes predictions.csv (id,category)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
USE_LEXICON = True
THRESHOLD = 0.70          # model must be at least this confident, else "unknown"
UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# 1. Cleaning
# --------------------------------------------------------------------------
PROVIDER = re.compile(r"^(PAYPAL\s*\*|KLARNA\s*\*|ZETTLE_?\s*\*|SUMUP\s*\*|IZ\s*\*|TRUSTLY\s*\*)\s*")
PREFIX = re.compile(r"^(KORTKOP)\s+\d{6}\s+")
LEAD_WORDS = re.compile(r"^((AUTOGIRO|BETALNING|BG|INSATTNING|UTTAG)\s+)+")
FOREIGN_TAIL = re.compile(r"\s+(EUR|DKK|NOK|USD|GBP)\s+[\d.,]+$")


FOLD = str.maketrans("ÅÄÖØÉ", "AAOOE")


def fold(text):
    """Upper-case and drop Swedish accents, so VÄSTTRAFIK == VASTTRAFIK."""
    return str(text).upper().translate(FOLD)


def upper(text):
    return fold(text).strip()


def core(text, keep_digits=False):
    """Merchant-only text: no dates, ids, amounts or payment-provider wrappers."""
    t = upper(text)
    t = PREFIX.sub("", t)
    t = FOREIGN_TAIL.sub("", t)
    t = PROVIDER.sub("", t)
    t = LEAD_WORDS.sub("", t)
    if not keep_digits:
        t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^A-ZÅÄÖØÉ0-9&. *]" if keep_digits else r"[^A-ZÅÄÖØÉ&. *]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# 2. Rules
# --------------------------------------------------------------------------
def _rx(words):
    """Whole-word match; a trailing * means 'starts with'."""
    parts = []
    for w in words:
        stem = w.endswith("*")
        w = fold(w.rstrip("*"))
        parts.append(r"(?<![A-ZÅÄÖØÉ0-9])" + re.escape(w) + ("" if stem else r"(?![A-ZÅÄÖØÉ0-9])"))
    return re.compile("|".join(parts))


# Ordered: first match wins. Specific categories go before broad ones.
LEXICON = [
    ("credit.collections", ["KRONOFOGD*", "INTRUM", "LINDORFF", "INKASSO*", "PAIR FINANCE", "MAREX", "CAB INKASSO"]),
    ("credit.loan", ["CSN", "KLARNA AB", "KLARNA", "QLIRO", "WALLEY", "ANYFIN", "SANTANDER*", "NORDAX*", "RESURS*",
                     "SEB KORT*", "KREDITKORT*", "AMERICAN EXPRESS", "AMEX", "BOLÅN*", "HYPOTEK*", "AMORTERING*", "LENDO",
                     "MARGINALEN*", "CETELEM", "IKANO*", "COLLECTOR*", "FÖRSTA KORT", "SVEA BANK*"]),
    ("spend.gambling", ["UNIBET", "BETSSON", "SVENSKA SPEL", "ATG", "BET365", "LEOVEGAS", "CASINO*", "BETWAY",
                        "KINDRED", "NORSK TIPPING", "LOTTO", "POKER*", "COMEON", "MR GREEN", "BETANO",
                        "BETSAFE", "NETENT", "SPELBOLAG*", "BINGO*", "VEIKKAUS", "NORDICBET", "SPEEDYBET",
                        "BETFAIR", "COOLBET", "EXPEKT"]),
    ("spend.family", ["BARNOMSORG*", "FÖRSKOLA*", "DAGIS", "SKOLA*", "SKOLAN", "FRITIDS*", "FRITIDSHEM*",
                      "PEDAGOG*", "BARNOMS*", "KOMVUX"]),
    ("spend.health", ["APOTEK*", "APOTEA", "KRY", "TANDVÅRD*", "TANDLÄKAR*", "VÅRDCENTRAL*", "LÄKAR*", "DOKTOR*",
                      "PRAKTIKERTJÄNST", "CAPIO", "FOLKTANDVÅRDEN", "VÅRDGIVAR*", "DOCTOR*", "PHARMACY", "PHARMA*",
                      "FARMACIA*", "MEDICIN*", "KIROPRAKT*", "FYSIO*", "SJUKHUS*", "DENTAL*", "1177", "LLOYDS APOTEK"]),
    ("spend.housing", ["ELLEVIO", "VATTENFALL*", "E.ON*", "FORTUM*", "TELIA*", "TELENOR*", "TELE2*", "COMHEM",
                       "BREDBAND*", "BOSTAD*", "BOSTÄDER*", "HEIMSTADEN*", "FASTIGHET*", "HYRA*", "FLYTT*",
                       "HYRESGÄST*", "BRF*", "BOSTADSRÄTT*", "ELNÄT*", "ELHANDEL*", "GÖTEBORG ENERGI", "TRE SVERIGE",
                       "HALEBOP", "COMVIQ", "TELIA", "BAHNHOF", "BOX BREDBAND", "FORTUM", "VATTEN*", "FJÄRRVÄRME*",
                       "RIKSBYGGEN", "HSB", "FAMILJEBOSTÄDER", "KRONOBERGSBYGGEN"]),
    ("spend.groceries", ["ICA", "WILLYS", "COOP", "HEMKÖP", "HEMKOP", "LIDL", "CITY GROSS", "NETTO", "REMA", "PINGO DOCE",
                         "MATHEM", "ALDI", "TESCO", "CARREFOUR", "MENY", "KIWI", "SUPERMARKET*", "LIVSMEDEL*",
                         "SUPERBRUGSEN", "FØTEX", "FOETEX", "BILKA", "IRMA", "KVICKLY",
                         "MERCADONA", "CONTINENTE", "AUCHAN", "INTERMARCHE", "E.LECLERC", "EDEKA", "REWE", "PRISMA",
                         "S-MARKET", "K-MARKET", "SUPERMERCADO*", "STORA COOP",
                         "AXFOOD"]),
    ("spend.dining", ["UBER *EATS", "UBER EATS", "FOODORA", "WOLT", "DELIVEROO", "MCDONALD*", "BURGER KING", "MAX",
                      "ESPRESSO HOUSE", "STARBUCKS", "SUBWAY", "KFC", "HESBURGER", "KAFE*", "CAFE*", "CAFÉ*", "KAFFE*",
                      "COFFEE*", "RESTAURANG*", "RESTAURANT*", "RESTAURANTE*", "PIZZ*", "SUSHI*", "KEBAB*", "BAGERI*",
                      "BAKERY", "PUB", "BISTRO*", "FOODTRUCK*", "FOOD TRUCK*", "SALLAD*", "TIME OUT MARKET", "CERVEJARIA*",
                      "FIKA*", "KONDITORI*", "GRILL*", "THAI*", "TAKEAWAY", "CATERING*", "BRASSERIE*", "TRATTORIA*",
                      "OSTERIA*", "PASTA*", "TACO*", "BURGER*", "STEAKHOUSE*", "VINBAR*", "WINE BAR",
                      "DONER*", "PIZZERIA*", "BRASSERI*", "SANDWICH*", "DELIKATESS*",
                      "ICE CREAM", "DUNKIN*", "NORDSTAN KAFE", "ESPRESSOHOUSE", "CAFEET",
                      "BAGEL*", "MATBAR*", "STRANDCAFE*", "KROG*", "HAMBURGER*", "FIKAROM*"]),
    ("spend.transport", ["UBER *TRIP", "UBER TRIP", "SL", "SKÅNETRAFIKEN*", "VÄSTTRAFIK*", "SJ", "MTR*", "VOI",
                         "TAXI*", "TAXI", "CIRCLE K", "OKQ*", "PREEM*", "ST1", "SHELL", "INGO",
                         "EASYPARK", "PARKERING*", "APCOA*", "Q-PARK*", "Q PARK*", "ØRESUNDSBRON", "ORESUNDSBRON",
                         "BROAVGIFT*", "TRAFIKVERKET*", "COMBOIOS*", "BUSS*", "FLIXBUS",
                         "TRAFIK*", "PARKING*", "TANKSTELLE*", "BENSIN*", "DRIVMEDEL*", "UNO-X", "UNOX",
                         "QSTAR", "Q8", "METRO*", "CARRIS*", "TRANSDEV*", "NETTBUSS*",
                         "ENTUR", "NSB", "DSB", "LYFT", "FREE NOW", "CABONLINE",
                         "TESLA SUPERCHARGER", "CHARGEFINDER",
                         "SUNDBY"]),
    ("spend.leisure", ["NETFLIX*", "SPOTIFY*", "VIAPLAY*", "HBO*", "DISNEY*", "YOUTUBE*", "STEAM*", "PLAYSTATION*",
                       "XBOX*", "NINTENDO*", "SATS", "FRISK*", "GYM*", "BIO", "SF BIO", "FILMSTADEN*", "TICKETMASTER*",
                       "LISEBERG*", "GRÖNA LUND*", "LEGOLAND*", "LALANDIA*", "SKANSEN*", "MUSEU*", "MUSEUM*",
                       "BOOKING.COM*", "HOTEL*", "HOTELL*", "SAS", "NORWEGIAN*", "RYANAIR*", "AIRBNB*", "FLYG*",
                       "STRAVA*", "APPLE MUSIC", "AUDIBLE*", "STORYTEL*", "BOKUS*", "TIVOLI*", "KOLMÅRDEN*", "FUNKIS*",
                       "EVENTIM*", "LIVE NATION*", "KONSERT*", "TEATER*", "THEATRE*", "CINEMA*", "KINO*", "BOWLING*",
                       "HOSTEL*", "RESORT*", "SKIPASS*", "EASYJET*", "LUFTHANSA*", "KLM", "FINNAIR*", "WIZZ*",
                       "TWITCH*", "EPIC GAMES*", "BLIZZARD*", "PRIME VIDEO", "DAZN*", "C MORE", "TV4 PLAY",
                       "TRIPADVISOR*", "EXPEDIA*", "HOTELS.COM*", "WATER PARK*", "PADEL*",
                       "SIMHALL*", "SKIDOR*", "GOLF*", "ÄVENTYR*", "FRILUFTS*", "SPELHALL*"]),
    ("spend.shopping", ["AMAZON*", "ZALANDO*", "H&M", "IKEA*", "JULA", "BAUHAUS*", "BYGGMAX*", "CLAS OHLSON*", "ELGIGANTEN*",
                        "MEDIAMARKT*", "KOMPLETT*", "WEBHALLEN*", "LYKO*", "KICKS*", "SEPHORA*", "JYSK*", "MIO", "ÅHLÉNS*",
                        "ZARA*", "ARKET", "BOOZT*", "NELLY*", "ASOS*", "SPORTAMORE*", "INTERSPORT*", "XXL", "DUSTIN*",
                        "POSTNORD*", "LINDEX*", "KAPPAHL*", "GINA TRICOT*", "WEEKDAY*", "STADIUM*", "NORDIC NEST*",
                        "BILTEMA*", "HORNBACH*", "BYGGOTEK*", "CDON*", "ADLIBRIS*",
                        "PRIMARK*", "UNIQLO*", "PULL&BEAR*", "BERSHKA*",
                        "MONKI*", "BUTIK*", "DECATHLON*",
                        "ALIEXPRESS*", "TEMU*", "SHEIN*", "ETSY*", "EBAY*", "APPLE.COM*",
                        "SYSTEMBOLAGET*", "SYSTEMET", "FLYING TIGER",
                        "OPTIK*", "SPECSAVERS*", "SYNOPTIK*", "INTERFLORA*",
                        "DOUGLAS*",
                        "HÅRSTUDIO*"]),
]
LEXICON = [(cat, _rx(words)) for cat, words in LEXICON]

# What a Swish message can say about the purpose of a payment to a person.
SWISH_MESSAGES = [
    ("spend.housing", ["HYRA*", "EL", "ELRÄKNING*", "INTERNET", "ÖVERLÅT*", "FLYTT*", "BOSTAD*"]),
    ("spend.dining", ["LUNCH*", "MIDDAG*", "FIKA*", "PIZZ*", "SUSHI*", "ÖL", "DRINKS", "BRUNCH*", "FRUKOST*", "RESTAURANG*"]),
    ("spend.transport", ["TAXI*", "BENSIN*", "PARKERING*", "RESA*"]),
    ("spend.leisure", ["GYM*", "BIO", "KONSERT*", "TRÄNING*", "PADEL*", "SPOTIFY*", "NETFLIX*"]),
    ("spend.groceries", ["MAT", "INKÖP*", "MATVAROR*", "HANDLAT*"]),
    ("spend.family", ["BARNPASS*", "FÖRSKOLA*", "SKOLA*", "DAGIS", "FRITIDS*"]),
    ("spend.health", ["APOTEK*", "LÄKAR*", "TANDLÄKAR*"]),
]
SWISH_MESSAGES = [(c, _rx(w)) for c, w in SWISH_MESSAGES]
SWISH_NAME = re.compile(r"^[A-ZÅÄÖÉ][A-ZÅÄÖÉ-]+( [A-ZÅÄÖÉ])?$")   # "AMIRA H" or "ANNA"
BUSINESS_SUFFIX = re.compile(r"\b(AB|HB|KB|AS|OY|APS|A/S|ENK)\b")

SAVINGS_WORDS = _rx(["SPAR*", "BUFFERT*", "SPARANDE*", "SPARKONTO*", "SPARKONTOT"])
TRANSFER_WORDS = re.compile(r"^(OVERFORING|OVERF\.?|TRANSFER|EGEN OVERF)\b")


def first_lexicon(text):
    for cat, rx in LEXICON:
        if rx.search(text):
            return cat
    return None


def savings_like(account_type, account_name, account=""):
    return str(account_type).lower() == "savings" or bool(SAVINGS_WORDS.search(upper(f"{account_name} {account}")))


def add_counterparts(df):
    """For each row, find the mirrored row on another own account (same customer, opposite amount, +-3 days)."""
    df = df.copy()
    df["pdate"] = pd.to_datetime(df["date"], errors="coerce")
    partner_savings = [None] * len(df)      # None = no partner found
    idx = {}
    for i, r in enumerate(df.itertuples(index=False)):
        idx.setdefault((r.customer, round(-float(r.amount_sek), 2)), []).append(i)
    rows = list(df.itertuples(index=False))
    for i, r in enumerate(rows):
        best = None
        for j in idx.get((r.customer, round(float(r.amount_sek), 2)), []):
            o = rows[j]
            if o.account == r.account or pd.isna(r.pdate) or pd.isna(o.pdate):
                continue
            gap = abs((o.pdate - r.pdate).days)
            if gap <= 3 and (best is None or gap < best[0]):
                best = (gap, o)
        if best:
            o = best[1]
            partner_savings[i] = savings_like(o.account_type, o.account_name, o.account)
    df["partner_savings"] = partner_savings
    return df


def rule_label(row):
    """Return a category for 'obvious' rows, otherwise None."""
    text = upper(row["text"])
    amount = float(row["amount_sek"])
    acc_type = str(row["account_type"]).lower()
    t = core(row["text"])

    # own-account moves --------------------------------------------------
    if acc_type == "credit_card" and ("INBETALNING" in text or "BETALNING TACK" in text):
        return "transfer.own"
    if TRANSFER_WORDS.search(text) or ("SPARKONTO" in text and "OVERF" in text):
        own_sav = savings_like(row["account_type"], row.get("account_name", ""), row.get("account", ""))
        partner = row.get("partner_savings", None)
        if own_sav or SAVINGS_WORDS.search(text) or partner is True:
            return "transfer.savings"
        if partner is False:
            return "transfer.own"
        return None

    # cash ---------------------------------------------------------------
    if re.search(r"\b(UTTAG|BANKOMAT|KONTANTUTTAG|ATM)\b", text):
        return "transfer.cash"

    # swish --------------------------------------------------------------
    if text.startswith("SWISH"):
        rest = re.sub(r"^SWISH\s+(BETALNING|INBETALNING|TILL|FRAN)?\s*", "", text).strip()
        if BUSINESS_SUFFIX.search(rest):
            return first_lexicon(rest) or UNKNOWN    # a shop paid by Swish -> category of the shop
        tokens = rest.split()
        if SWISH_NAME.match(rest):
            return "transfer.person"                  # name only, no message
        # name followed by a message
        if len(tokens) >= 2:
            msg = " ".join(tokens[1:] if len(tokens[1]) > 1 else tokens[2:])
            for cat, rx in SWISH_MESSAGES:
                if rx.search(msg):
                    return cat
        return UNKNOWN

    # income -------------------------------------------------------------
    if amount > 0:
        if re.search(r"^LON\b|\bLON\b|SALARY|LONEUTBETALNING", text):
            return "income.employment"
        if re.search(r"FORSAKRINGSKASSAN|PENSIONSMYNDIGHETEN|BARNBIDRAG|ARBETSFORMEDLINGEN|SKATTEVERKET", text):
            return "income.public"
        if re.search(r"\b(RANTA|UTDELNING|ARSRANTA|KAPITALAVKASTNING)\b", text):
            return "income.capital"
        if re.search(r"TRADERA|BLOCKET|VINTED|SELLPY", text):
            return "income.sales"
        if re.search(r"^REGION\b|^KOMMUN\b|\bKOMMUN$", t):
            return "income.employment"

    # merchant lexicon (refunds/payouts take the category of what they reverse)
    return first_lexicon(core(row["text"], keep_digits=True)) if USE_LEXICON else None


# --------------------------------------------------------------------------
# 3. Small model
# --------------------------------------------------------------------------
def meta_tokens(df):
    out = []
    for r in df.itertuples(index=False):
        a = float(r.amount_sek)
        bucket = "b0" if abs(a) < 100 else "b1" if abs(a) < 500 else "b2" if abs(a) < 2000 else "b3" if abs(a) < 10000 else "b4"
        out.append(f"{'pos' if a > 0 else 'neg'} {str(r.account_type).lower()} {bucket}")
    return out


def make_model():
    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "core"),
        ("meta", CountVectorizer(token_pattern=r"\S+"), "meta"),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=20, max_iter=2000))])


def featurize(df):
    f = pd.DataFrame({"core": df["text"].map(core), "meta": meta_tokens(df)})
    return f


def fit(train):
    m = make_model()
    m.fit(featurize(train), train["category"])
    return m


def predict(df, model, threshold=THRESHOLD):
    df = add_counterparts(df)
    labels, source = [], []
    probs = model.predict_proba(featurize(df))
    classes = model.classes_
    for k, (_, row) in enumerate(df.iterrows()):
        lab = rule_label(row)
        if lab is not None:
            labels.append(lab)
            source.append("rule")
            continue
        p = probs[k]
        best = int(np.argmax(p))
        if p[best] >= threshold:
            labels.append(classes[best])
            source.append("model")
        else:
            labels.append(UNKNOWN)
            source.append("unknown")
    return labels, source


def main():
    test_path = sys.argv[1]
    train_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "transactions.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for col, default in [("account_name", ""), ("account", ""), ("customer", "c"), ("account_type", "checking")]:
        if col not in test.columns:
            test[col] = default
    model = fit(train)
    labels, _ = predict(test, model)
    out = pd.DataFrame({"id": test["id"], "category": labels})
    out.to_csv("predictions.csv", index=False)
    print(f"wrote predictions.csv ({len(out)} rows, {sum(l == UNKNOWN for l in labels)} unknown)")


if __name__ == "__main__":
    main()
