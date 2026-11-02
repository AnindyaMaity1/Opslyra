# Independent ticket annotation guide

Opslyra currently has no independently verified operational labels.
Its supplied ticket labels are demo data, and Cohen's kappa is therefore
**not measured**. Use this guide only when a real annotation study has
been approved and can be conducted with appropriate privacy controls.

## Preparation

1. Select a ticket sample and define the label taxonomy before annotation.
   Record the sampling method, date, and inclusion/exclusion criteria.
2. Provide both annotators with the same written definitions and examples.
   Include explicit rules for ambiguous, insufficient-information, and
   out-of-scope tickets.
3. Keep the two annotation files separate. Annotators must label
   independently and must not see each other's labels before the
   agreement calculation.
4. Use the same non-identifying `Ticket ID` in both files to align rows.
   Do not include names, email addresses, ticket text, or other personal
   data unless the study has an approved, secured handling process.

## CSV format

Each file must contain these columns:

```csv
Ticket ID,Ticket Type,Ticket Priority
sample-001,Billing inquiry,Low
sample-002,Technical issue,High
```

Use the same allowed category values in both files. Leave a label blank
when the annotator cannot assign it under the agreed codebook; pairs
with either label missing are excluded for that target. Duplicate IDs
are rejected, and only IDs present in both files are compared.

## Agreement calculation

From the repository root, run:

```powershell
python experiments.py --output instance/experiments kappa annotator_a.csv annotator_b.csv
```

The command writes `instance/experiments/kappa.json` with the overlapping
sample size, excluded unlabeled pairs, class counts, confusion tables,
observed and expected agreement, and Cohen's kappa for each configured
label column. Review the overlap and class counts before interpreting
kappa. A low-prevalence or highly imbalanced label can make kappa
misleading; report observed agreement and the confusion tables alongside
it.

Resolve disagreements only after preserving the independent labels and
the initial agreement result. If an adjudicated dataset is produced,
keep it as a separate version and document who adjudicated it and how.
Never replace or alter the original annotations to improve agreement.

## Reporting

Report the sample selection, annotator independence, codebook version,
missing-label exclusions, overlap count, per-label agreement, and any
adjudication procedure. Do not claim inter-annotator reliability until
the calculation has been run on genuine independent annotations.
