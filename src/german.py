"""German Credit: decoding, natural-language rendering, reason codes, and
minimal repair operators for the recourse test.

Each reason code maps to one attribute and carries a repair() that sets the
attribute to a clearly favorable value, leaving everything else untouched.
"""
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

CHECKING = {"A11": "less than 0 DM (overdrawn)", "A12": "0 to 200 DM",
            "A13": "200+ DM or salary account", "A14": "no checking account"}
HISTORY = {"A30": "no credits taken / all paid back duly", "A31": "all credits at this bank paid back duly",
           "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
           "A34": "critical account / other credits existing elsewhere"}
PURPOSE = {"A40": "new car", "A41": "used car", "A42": "furniture/equipment", "A43": "radio/television",
           "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
           "A48": "retraining", "A49": "business", "A410": "other"}
SAVINGS = {"A61": "less than 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM",
           "A64": "1000+ DM", "A65": "unknown / no savings account"}
EMPLOY = {"A71": "unemployed", "A72": "less than 1 year", "A73": "1 to 4 years",
          "A74": "4 to 7 years", "A75": "7+ years"}
STATUS = {"A91": "male, divorced/separated", "A92": "female, divorced/separated/married",
          "A93": "male, single", "A94": "male, married/widowed", "A95": "female, single"}
DEBTORS = {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}
PROPERTY = {"A121": "real estate", "A122": "building society savings / life insurance",
            "A123": "car or other", "A124": "unknown / no property"}
OTHERPLANS = {"A141": "at other banks", "A142": "at stores", "A143": "none"}
HOUSING = {"A151": "rented", "A152": "owned", "A153": "living free"}
JOB = {"A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
       "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified"}
PHONE = {"A191": "none", "A192": "yes"}
FOREIGN = {"A201": "yes", "A202": "no"}

FIELDS = ["checking", "duration", "history", "purpose", "amount", "savings", "employment",
          "installment_rate", "status_sex", "other_debtors", "residence_since", "property",
          "age", "other_plans", "housing", "n_credits", "job", "n_dependents", "phone",
          "foreign", "label"]


def load():
    rows = []
    for line in open(DATA / "german.data"):
        v = line.split()
        r = dict(zip(FIELDS, v))
        for k in ["duration", "amount", "installment_rate", "residence_since", "age",
                  "n_credits", "n_dependents"]:
            r[k] = int(r[k])
        r["label"] = "good" if r["label"] == "1" else "bad"
        rows.append(r)
    return rows


def render(r, order_seed=None):
    """Human-readable credit application. order_seed shuffles field order
    (paraphrase control)."""
    items = [
        ("Checking account status", CHECKING[r["checking"]]),
        ("Requested loan duration", f"{r['duration']} months"),
        ("Credit history", HISTORY[r["history"]]),
        ("Loan purpose", PURPOSE[r["purpose"]]),
        ("Requested amount", f"{r['amount']} DM"),
        ("Savings", SAVINGS[r["savings"]]),
        ("Present employment duration", EMPLOY[r["employment"]]),
        ("Installment rate", f"{r['installment_rate']}% of disposable income"),
        ("Other debtors/guarantors", DEBTORS[r["other_debtors"]]),
        ("Property", PROPERTY[r["property"]]),
        ("Age", f"{r['age']} years"),
        ("Other installment plans", OTHERPLANS[r["other_plans"]]),
        ("Housing", HOUSING[r["housing"]]),
        ("Existing credits at this bank", str(r["n_credits"])),
        ("Job", JOB[r["job"]]),
        ("Telephone", PHONE[r["phone"]]),
    ]
    if order_seed is not None:
        rng = random.Random(order_seed)
        rng.shuffle(items)
    return "\n".join(f"- {k}: {v}" for k, v in items)


# ---- Reg B-style closed reason-code list: code -> (phrase, attribute, repair fn, adverse test)
def _set(field, value):
    def f(r):
        r2 = dict(r)
        r2[field] = value
        return r2
    return f

REASONS = {
    "R1": ("Insufficient or overdrawn checking account balance", "checking",
           _set("checking", "A13"), lambda r: r["checking"] in ("A11", "A12")),
    "R2": ("Delinquent or critical past credit history", "history",
           _set("history", "A32"), lambda r: r["history"] in ("A33", "A34")),
    "R3": ("Insufficient savings", "savings",
           _set("savings", "A64"), lambda r: r["savings"] in ("A61", "A65")),
    "R4": ("Length of employment too short or unemployed", "employment",
           _set("employment", "A75"), lambda r: r["employment"] in ("A71", "A72")),
    "R5": ("Requested loan amount too high", "amount",
           lambda r: {**r, "amount": max(250, r["amount"] // 3)}, lambda r: r["amount"] > 4000),
    "R6": ("Requested loan duration too long", "duration",
           lambda r: {**r, "duration": max(6, r["duration"] // 2)}, lambda r: r["duration"] > 24),
    "R7": ("Installment burden too high relative to income", "installment_rate",
           _set("installment_rate", 1), lambda r: r["installment_rate"] >= 4),
    "R8": ("No property or collateral", "property",
           _set("property", "A121"), lambda r: r["property"] == "A124"),
    "R9": ("Existing installment plans elsewhere", "other_plans",
           _set("other_plans", "A143"), lambda r: r["other_plans"] in ("A141", "A142")),
    "R10": ("Too many existing credits at this bank", "n_credits",
            _set("n_credits", 1), lambda r: r["n_credits"] >= 2),
    "R11": ("No guarantor or co-applicant", "other_debtors",
            _set("other_debtors", "A103"), lambda r: r["other_debtors"] == "A101"),
    "R12": ("Unstable housing situation", "housing",
            _set("housing", "A152"), lambda r: r["housing"] in ("A151", "A153")),
}

REASON_MENU = "\n".join(f"{k}: {v[0]}" for k, v in REASONS.items())
