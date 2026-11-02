# Opslyra

This project is built under the guidance of **Prof. Dr. Govind Pandey**.

LinkedIn Profile - [Link](https://www.linkedin.com/in/dr-govind-pandey-145a43131/)

**From Every Ticket to Intelligent Action.**

Opslyra is a **SaaS-style IT service intelligence workspace** for exploring
support tickets, evaluating text-classification models, and running
human-controlled AI-assisted workflows. Its dashboard brings ticket
operations, model diagnostics, incident research, and data-quality checks into
one cohesive interface.

> **Deployment scope:** Opslyra currently runs as a local, single-workspace
> prototype. It is not yet a hosted, authenticated, multi-tenant SaaS service;
> it does not connect to a live ticketing system or send customer replies.
> The interface demonstrates a SaaS product experience, while persistence and
> computation remain local. See [limitations](#limitations) before using it
> with sensitive data.

## Screenshots

The screenshots show the application UI; the underlying results are generated
from the configured CSV at runtime.

| Overview | Graphs |
|---|---|
| ![Opslyra overview dashboard](docs/screenshots/dashboard.png) | ![Grouped cross-validation and label-signal graphs](docs/screenshots/graphs.png) |

| Adaptive AI | Research modules |
|---|---|
| ![Adaptive AI risk-coverage controls and preview](docs/screenshots/adaptive-ai.png) | ![Research modules](docs/screenshots/research.png) |

## Results on the Bundled Dataset

The following figures were recomputed from the checked-in
`customer_support_tickets.csv` using the current code. The comparison uses
three-fold `StratifiedGroupKFold`; normalized identical ticket text stays in
one fold. The majority baseline is fitted separately on each training fold.

| Prediction target | Labeled rows | Majority accuracy | TF-IDF + Logistic Regression | Difference |
|---|---:|---:|---:|---:|
| Ticket Priority | 8,469 | 25.88% | 24.06% | -1.82 percentage points |
| Ticket Status | 8,469 | 34.02% | 33.85% | -0.17 percentage points |

Neither target beats the majority baseline in this signal check. These results
do not establish that the labels were randomly generated; they show that this
text baseline does not provide an accuracy improvement under this evaluation.

| Dataset audit | Current result |
|---|---:|
| Rows / expected columns | 8,469 / 17 |
| Repeated rows by normalized ticket text | 144 |
| Exact-text groups with at least one conflicting type, priority, or status | 104 |
| Near-duplicate pairs at TF-IDF cosine similarity >= 0.92 | 6 of 400 sampled distinct texts |

The near-duplicate scan is a deterministic sample, not a census. Of its six
pairs, five had disjoint priority labels and two had disjoint status labels.
The exact-text conflicts and weak baseline results are data-quality warnings,
not proof that labels are incorrect.

Results depend on the dataset and software versions. Run the audit and
experiments again after changing the CSV; do not present these figures as
future or production performance.

## SaaS Workspace Features

Opslyra is organized as a single analyst workspace with a persistent sidebar,
workspace controls, responsive cards, tables, and charts. The main navigation
contains the following product areas:

| Workspace area | What it provides |
|---|---|
| **Overview** (`/`) | A high-level view of the configured ticket dataset and service signals. |
| **Graphs** (`/graphs`) | Cross-validated label-signal comparisons, duplicate-text audits, probability calibration, and downloadable charts. |
| **Tickets** (`/tickets`) | Search and filter ticket records, browse the dataset in pages of 10, and open a detailed ticket view. |
| **AI prediction** (`/predict`) | Submit a ticket subject and description to estimate ticket type and priority using the trained local baselines. |
| **Model evaluation** (`/models`) | Review validation metrics and class-level performance, and train refreshed baseline models. |
| **Incident lab** (`/intelligence`) | Explore ten local workflows for human review, novelty, relationships, routing simulations, and model operations. |
| **Adaptive AI** (`/adaptive-ai`) | Configure workspace and ticket-level assistance modes, approval policies, and validation-informed risk–coverage settings. |
| **Research modules** (`/research`) | Run open-set and supporting experiments with grouped splits and reported evaluation details. |
| **Data quality** (`/data-quality`) | Inspect schema, missingness, timestamp parseability, and observed label distributions without rewriting source values. |

### Ticket and model workflows

- **Ticket operations:** search/filter the configured CSV, review core ticket
  fields in a compact paginated table, and open a full-width ticket detail
  view. The detail page separates the ticket description from its attributes
  and shows the activity timeline when local AI workflow events exist.
- **Label overlays:** human-approved ticket type and priority changes are
  stored as local overlays. They do not modify the source CSV. Ticket pages
  identify active human takeover and display saved overlay labels.
- **Predictions:** the AI prediction page accepts subject and description text,
  redacts common sensitive patterns before model use, and presents model
  estimates as suggestions—not verified labels.
- **Evaluation:** ticket type and ticket priority are evaluated separately.
  Reports include repeated, grouped cross-validation, accuracy, macro/weighted
  F1, descriptive intervals, class performance, and calibration diagnostics
  where available.
- **Chart export:** Graphs include cross-validated accuracy versus a
  fold-local majority baseline, exact/near-duplicate label-conflict analysis,
  and probability reliability diagnostics. Supported chart downloads are PNG
  files tagged with 1200-DPI resolution metadata.

### Adaptive AI operating modes

Adaptive AI makes the allowed level of assistance visible and configurable:

- **Manual:** no AI workflow actions are taken.
- **Copilot:** recommendations require a human decision.
- **Selective automation:** local label actions can be gated by a chosen
  validation risk–coverage operating point.
- **Autonomous:** permits only the explicitly enabled local label actions;
  it does not call external systems or contact customers.
- **Human takeover:** ticket-specific pause for AI recommendations until
  control is released.

The policy can enable category and priority actions and choose between a
target-coverage or target-selective-risk operating point. Dry-run previews are
designed to show proposed workflow results without committing state changes.
Mode changes, recommendations, approvals, corrections, and takeover events
are recorded in the local SQLite audit trail. The policy and risk thresholds
are demo controls, not service-level guarantees.

### Incident Lab workflows

The Incident Lab (`/intelligence`) contains ten independent local explorations:

1. **Human-in-the-loop learning** — save reviewed labels, rank corrections by
   information value, and train a model version from available corrections.
2. **Open-set incident discovery** — compare submitted text with observed
   tickets and show nearest examples and threshold-based novelty signals.
3. **Counterfactual text explorer** — compare model outputs under changed
   wording; this is a sensitivity demonstration, not causal evidence.
4. **Incident relationship graph** — surface related tickets using text and
   available product, error-code, and time clues.
5. **SLA risk sandbox** — adjust assumptions and inspect a transparent
   simulated risk index.
6. **Adaptive routing simulator** — compare skill matching with a
   round-robin baseline under configured team constraints.
7. **Recurrence and early-warning radar** — inspect observed ticket volumes
   and text clusters; forecasts are only meaningful when date support allows.
8. **Privacy-aware text handling** — scan submitted text for common sensitive
   patterns and retain a local count-oriented audit.
9. **Evidence-grounded resolution memory** — retrieve similar historical
   tickets that have recorded resolutions.
10. **Model reliability observatory** — inspect model versions and available
    held-out or chronological evaluation signals.

Routing capacity, team skills, processing time, SLA budgets, and text-keyword
impact are configured or proxy values where the dataset provides no verified
operational ground truth. Treat these tools as simulations.

### Research modules

The primary experiment in Research (`/research`) is **open-set ticket
classification and unseen-class rejection**. It supports selectable targets,
unknown classes, repeatable random seeds, grouped duplicate handling,
validation-based threshold selection, per-seed metrics, and ROC/precision–
recall visualizations.

Supporting research modules cover:

- Dataset and label-signal audits, baseline comparisons, calibration, feature
  ablation, and risk–coverage analysis.
- A novelty-aware annotation queue combining model uncertainty, novelty,
  diversity, text-derived impact, and estimated annotation effort.
- A constrained routing simulator separating text-model recommendations from
  configured team workload, capacity, and SLA feasibility.
- Temporal incident fingerprints that combine ticket language and available
  metadata, with recurrence evaluation only when timestamps and sample sizes
  support it.

Research outputs are exploratory diagnostics. The app does not claim
production precision/recall for incident detection, verified operational
routing outcomes, or causal impact.

## Product Architecture and Local Data Flow

1. **Load:** Flask reads the configured CSV through the dataset layer and
   validates the required ticket schema.
2. **Prepare:** model text is built from permitted ticket text fields and
   passed through pattern-based redaction before supported prediction and
   analysis paths.
3. **Analyze:** pandas/NumPy/scikit-learn power the dashboard summaries,
   grouped evaluation, graphs, predictions, and research modules.
4. **Persist locally:** SQLite stores workspace settings, review events,
   activity, and approved label overlays. Model artifacts and experiment
   snapshots are stored under `instance/`.
5. **Present:** Flask templates render the single-workspace SaaS-style UI;
   static CSS and JavaScript provide the responsive interface and chart
   interactions.

The input CSV is treated as the source dataset and is not edited when a human
approves a label correction. SQLite state and model artifacts are local to
the current installation; they are not user-isolated or synchronized across
hosted workspaces.

## Research and Reproducibility

### Evaluation design

The interactive model reports use repeated stratified, exact-text-grouped
cross-validation across multiple seeds, descriptive confidence intervals,
and a separate holdout when class support permits. A chronological
train/test comparison is reported only when timestamps are explicitly
trusted and sufficient data is available. The final model is fitted on
eligible data for prediction; this is separate from the evaluation estimates.

The Research baseline comparison includes majority class, Complement Naive
Bayes, linear SVM, TF-IDF + Logistic Regression, and an embedding + Logistic
Regression path. Sentence-transformer embeddings run only when the optional
package and `OPSLYRA_EMBEDDING_MODEL` are configured; otherwise the embedding
path uses TF-IDF + TruncatedSVD.

### CLI

Python 3.10+ is supported through the compatibility requirements. The pinned
test environment uses Python 3.13.1. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python experiments.py --output instance/experiments --seeds 0,1,2 --folds 3 --max-rows 800 reproduce
```

If GNU Make is installed, the same standard experiment suite can be run with:

```sh
make reproduce
```

Make variables can be overridden, for example:

```sh
make reproduce PYTHON=python SEEDS=0,1,2 MAX_ROWS=800
```

The suite writes dataset audit, model evaluation, calibration/ablation,
open-set, and risk–coverage JSON records under `instance/experiments/`.
Individual commands are `audit`, `evaluate`, `ablation`, `openset`,
`risk-coverage`, and `kappa`. Options go before the subcommand. For example:

```powershell
python experiments.py --output instance/experiments --seeds 0,1,2 --unknown "Refund request" openset
```

Independent label agreement is not currently measured. To calculate Cohen's
kappa from two independently produced annotation CSVs, see
[the annotation guide](docs/ANNOTATION_GUIDE.md). Do not treat the bundled
labels as independently verified operational ground truth.

## Dataset and Provenance

The bundled CSV has 8,469 rows and 17 required columns, including ticket
subject, description, type, status, and priority. The project identifies the
file as sourced from Kaggle. The exact Kaggle listing, citation, license,
retrieval date, and independent verification of labels/timestamps have not
been recorded. Check the original Kaggle terms before redistribution or reuse;
the repository's MIT license does **not** license the dataset.

Missing values remain missing in the raw and analysis data. They are not
replaced with invented labels. The current CSV has 5,700 missing values in
each of `Resolution`, `Time to Resolution`, and `Customer Satisfaction
Rating`, and 2,819 in `First Response Time`. Counts should be recomputed if
the dataset changes.

To use another CSV, set `TICKETMIND_DATASET` to its path. Include all required
columns. A provenance sidecar named `<csv-filename>.provenance.json` should
record source, citation, license, retrieval date, and label/timestamp
verification status. See the example in [Configuration](#configuration).

## Installation and Run

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:OPSLYRA_SECRET_KEY = "use-a-long-random-secret"
python app.py
```

Open <http://127.0.0.1:5000>. To use a different CSV:

```powershell
$env:TICKETMIND_DATASET = "C:\path\to\tickets.csv"
python app.py
```

On other platforms, create/activate a virtual environment and install the
appropriate requirements file. Use `requirements-lock.txt` for the pinned
Python 3.13.1 environment; use `requirements.txt` for the supported
compatibility range.

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TICKETMIND_DATASET` | Input CSV | `customer_support_tickets.csv` beside `app.py` |
| `OPSLYRA_SECRET_KEY` | Flask session signing key | Random per process if unset |
| `HOST` | Development server bind address | `127.0.0.1` |
| `PORT` | Development server port | `5000` |
| `FLASK_DEBUG` | Enable Flask debug mode when set to `true` | Disabled |

For alternate datasets, place a JSON file beside the CSV with `.provenance.json`
appended to its filename:

```json
{
  "source_kind": "external",
  "source": "Dataset source name",
  "label": "External dataset",
  "citation": "Publication or source URL",
  "license": "License identifier or pending review",
  "retrieval_date": "YYYY-MM-DD",
  "labels_verified": false,
  "timestamps_trustworthy": false
}
```

Only mark labels or timestamps verified when the supporting evidence has been
independently checked and documented.

## Tests and CI

Run the test suite locally:

```powershell
python -m pytest -q
```

GitHub Actions runs pytest on Python 3.10 using `requirements.txt` and Python
3.13.1 using the exact pins in `requirements-lock.txt`. The workflow is
`.github/workflows/ci.yml`.

## CSRF Protection

All unsafe HTTP methods are checked by a global Flask request hook against a
session-bound CSRF token. HTML forms submit the token as a form field, and
JSON requests send it in the `X-CSRF-Token` header. Requests without a valid
token are rejected with HTTP 403. This is a local prototype safeguard; it is
not authentication or authorization.

## Technology

- Python, Flask, SQLite
- pandas, NumPy, SciPy, scikit-learn
- HTML, CSS, JavaScript, and inline SVG/chart visualizations
- pytest and GitHub Actions

## Limitations

- This is a single-user local research prototype, not an authenticated
  multi-user service or production ticketing system.
- Dataset identity is attributed to Kaggle by the project owner, but its
  exact source citation, terms, and label verification are not recorded.
- Confidence scores are not automatically calibrated. Risk–coverage points
  are validation diagnostics and do not guarantee future risk.
- Open-set held-out classes are proxies for unseen incidents, not verified
  out-of-distribution production events.
- The ticket dataset has no assigned-team ground truth or verified SLA-breach
  labels; routing and SLA outcomes are simulations.
- Available response-time dates are insufficient for reliable recurrence
  forecasting.
- Redaction uses patterns and does not guarantee removal of all personal or
  sensitive information.

## License

Opslyra source code and project documentation are licensed under the MIT
License; see [LICENSE](LICENSE). The dataset and any third-party materials
are excluded from that grant and remain subject to their own terms.

---

**Opslyra — From Every Ticket to Intelligent Action.**
