# Opslyra

This project is built under the guidance of **Prof. Dr. Govind Pandey**.
Department of CSE (AI & ML), Dayananda Sagar University, Bengaluru, Karnataka 562112, India.

LinkedIn Profile - [Link](https://www.linkedin.com/in/dr-govind-pandey-145a43131/)

**From Every Ticket to Intelligent Action.**

Opslyra is a **SaaS-style IT service intelligence workspace** for exploring
customer-support tickets, evaluating classification models, and testing
human-controlled AI-assisted workflows. It brings ticket operations, model
quality, incident exploration, and research tools into one analyst-friendly
dashboard.

> **Current deployment scope:** Opslyra is a local, single-workspace
> demonstration and research application. It is not a hosted, authenticated,
> multi-tenant SaaS service and does not connect to a live ticketing system or
> send customer replies. The UI demonstrates a SaaS product experience; data,
> state, and model artifacts remain local. See [Important limitations](#important-limitations)
> before using it with sensitive data.

## Book Index

- [Opslyra Overview](#opslyra-overview)
  - [Operations Dashboard](#operations-dashboard)
  - [Ticket Workspace and Graphs](#ticket-workspace-and-graphs)
  - [Adaptive AI and Research Lab](#adaptive-ai-and-research-lab)
- [What the Platform Does](#what-the-platform-does)
  - [Workspace Navigation](#workspace-navigation)
- [Adaptive AI Modes](#adaptive-ai-modes)
- [Incident Lab](#incident-lab)
- [Research Lab](#research-lab)
- [How It Works](#how-it-works)
- [Results on the Bundled Dataset](#results-on-the-bundled-dataset)
- [Research and Reproducibility](#research-and-reproducibility)
  - [Reproducible CLI Experiments](#reproducible-cli-experiments)
- [Installation and Run](#installation-and-run)
  - [Windows PowerShell](#windows-powershell)
  - [macOS or Linux](#macos-or-linux)
  - [Configuration](#configuration)
- [Tests and CI](#tests-and-ci)
- [Technology](#technology)
- [Important Limitations](#important-limitations)
- [Detailed Product Guide](#detailed-product-guide)
- [HTTP Route Reference](#http-route-reference)
- [Dataset Schema Reference](#dataset-schema-reference)
- [Modeling and Evaluation Reference](#modeling-and-evaluation-reference)
- [Local Storage and State](#local-storage-and-state)
- [Privacy and Security Notes](#privacy-and-security-notes)
- [Troubleshooting](#troubleshooting)
- [Development Guide](#development-guide)
- [SaaS Productization Roadmap](#saas-productization-roadmap)
- [Glossary](#glossary)
- [FAQ](#faq)
- [License](#license)

## Opslyra Overview

### Operations Dashboard

The dashboard presents an at-a-glance view of the configured support-ticket
dataset and operational signals.

It prioritizes open Critical and High tickets, summarizes open and waiting
work, reports the average of available satisfaction ratings, and displays
ticket-type and priority distributions.

### Ticket Workspace and Graphs

Search and filter tickets, inspect individual records, and review label-signal
and model-diagnostic charts.

The ticket workspace supports exact category/status/priority filters and
literal text search across ticket ID, subject, and description. Graphs compare
text-model accuracy with a majority baseline, inspect confidence calibration,
and report exact and sampled near-duplicate label conflicts.

### Adaptive AI and Research Lab

The Adaptive AI workspace makes assistance modes and human approval controls
visible. Research modules provide repeatable open-set experiments and
supporting dataset, model, active-learning, routing, and temporal analyses.

## What the Platform Does

- **Operations overview:** summarize the configured ticket dataset and service
  signals in a dashboard.
- **Ticket management:** search and filter the CSV-backed ticket list, browse
  10 records per page, and inspect ticket descriptions, attributes, and local
  workflow history.
- **AI prediction:** estimate ticket type and priority from a subject and
  description using local text-classification models.
- **Model evaluation:** review repeated grouped cross-validation, accuracy,
  macro/weighted F1, descriptive confidence intervals, and per-class
  performance.
- **Graphs:** compare cross-validated model accuracy against a fold-local
  majority baseline, inspect duplicate-text label conflicts, and review
  probability calibration. Chart downloads are PNGs with 1200-DPI resolution
  metadata.
- **Adaptive AI controls:** select manual, copilot, selective, or autonomous
  workspace modes; use ticket-specific human takeover; and configure
  validation-informed risk–coverage policies.
- **Human review and label overlays:** save reviewed corrections and approved
  labels locally without changing the source CSV.
- **Incident Lab:** explore ten local workflows spanning novelty, wording
  sensitivity, related tickets, simulated SLA/routing, privacy-aware text
  handling, resolution retrieval, and model history.
- **Research Lab:** run open-set evaluation as the primary experiment, with
  supporting dataset audits, baselines, active-learning, routing, temporal,
  calibration, ablation, and risk–coverage modules.
- **Data quality:** inspect schema, missing values, timestamp parseability,
  and observed label distributions without rewriting source values.
- **Local audit and experiment records:** persist workspace settings, review
  activity, label overlays, model artifacts, and experiment snapshots locally.

### Workspace Navigation

| Area | Route | Purpose |
|---|---|---|
| Overview | `/` | Summarize the configured dataset and service signals. |
| Graphs | `/graphs` | View label-signal, duplicate-text, and calibration diagnostics. |
| Tickets | `/tickets` | Search, filter, and browse ticket records. |
| AI prediction | `/predict` | Estimate ticket type and priority from ticket text. |
| Model evaluation | `/models` | Review model metrics and train refreshed baselines. |
| Incident Lab | `/intelligence` | Run local incident and workflow explorations. |
| Adaptive AI | `/adaptive-ai` | Configure assistance modes, policy, and risk–coverage settings. |
| Research modules | `/research` | Run open-set and supporting research experiments. |
| Data quality | `/data-quality` | Inspect data schema, missingness, dates, and labels. |

## Adaptive AI Modes

Adaptive AI is designed to make the level of automation explicit:

- **Manual:** no AI workflow actions are taken.
- **Copilot:** recommendations require a human decision.
- **Selective automation:** local label actions can be gated by a selected
  validation risk–coverage operating point.
- **Autonomous:** permits only the explicitly enabled local category and
  priority actions; it does not contact customers or external systems.
- **Human takeover:** pauses AI recommendations for an individual ticket
  until human control is released.

The policy supports target coverage or target selective risk settings. Dry-run
previews are intended to show proposed results without committing workflow
state. Mode changes, recommendations, approvals, corrections, and takeover
events are recorded in the local SQLite audit trail. The policy is a
demonstration control, not a service-level or future-risk guarantee.

## Incident Lab

The Incident Lab (`/intelligence`) includes ten local workflows:

1. **Human-in-the-loop learning:** save reviewed labels, rank corrections by
   information value, and train a model version using available corrections.
2. **Open-set incident discovery:** compare submitted text with observed
   tickets and show nearest examples and threshold-based novelty signals.
3. **Counterfactual text explorer:** compare outputs under changed wording.
   This demonstrates model sensitivity, not causal impact.
4. **Incident relationship graph:** show related tickets using text and
   available product, error-code, and time clues.
5. **SLA risk sandbox:** adjust assumptions and inspect a transparent
   simulated risk index.
6. **Adaptive routing simulator:** compare skill matching with a round-robin
   baseline under configured team constraints.
7. **Recurrence and early-warning radar:** inspect observed ticket volumes and
   text clusters when available date support is sufficient.
8. **Privacy-aware text handling:** scan text for common sensitive patterns
   and retain a local count-oriented audit.
9. **Evidence-grounded resolution memory:** retrieve similar historical
   tickets with recorded resolutions.
10. **Model reliability observatory:** inspect model versions and available
    holdout or chronological evaluation signals.

Team skills, workload, capacity, processing times, SLA budgets, and
text-keyword impact may be configured values or proxies. The bundled dataset
does not provide verified operational ground truth for routing or SLA
outcomes; treat these workflows as simulations.

## Research Lab

The primary Research experiment is **open-set ticket classification and
unseen-class rejection**. It supports selectable targets and unknown classes,
repeatable random seeds, duplicate-group-aware splits, validation-based
threshold selection, per-seed metrics, and ROC/precision–recall curves.

Supporting research tools include:

- Dataset and label-signal audits, baseline comparisons, calibration,
  feature ablation, and risk–coverage analysis.
- A novelty-aware annotation queue using uncertainty, novelty, diversity,
  text-derived impact, and estimated annotation effort.
- A constrained routing simulator separating text-model recommendations from
  configured team workload, capacity, and simulated SLA feasibility.
- Temporal incident fingerprints using ticket language and available
  metadata, with recurrence evaluation only when dates and sample sizes allow.

Research results are exploratory diagnostics. The app does not claim
production incident-detection precision/recall, verified routing outcomes, or
causal impact.

## How It Works

1. **Load data:** Flask reads the configured CSV and validates the expected
   ticket schema.
2. **Prepare ticket text:** supported model and analysis paths use ticket text
   fields and pattern-based redaction for common sensitive values.
3. **Analyze:** pandas, NumPy, and scikit-learn power the dashboard, text
   models, evaluations, charts, and research workflows.
4. **Save local workspace state:** SQLite stores settings, activity, review
   events, and approved label overlays. Model artifacts and experiment
   snapshots are saved under `instance/`.
5. **Render the workspace:** Flask templates, CSS, JavaScript, and inline SVG
   provide the responsive SaaS-style interface.

Human-approved labels are stored as local overlays; the source CSV remains
unchanged. Local state is not isolated by user or synchronized across hosted
workspaces.

## Results on the Bundled Dataset

These figures were recomputed from the checked-in
`customer_support_tickets.csv`. The signal check uses three-fold
`StratifiedGroupKFold`; normalized identical ticket text stays in one fold,
and the majority baseline is fitted independently on each training fold.

| Prediction target | Labeled rows | Majority accuracy | TF-IDF + Logistic Regression | Difference |
|---|---:|---:|---:|---:|
| Ticket Priority | 8,469 | 25.88% | 24.06% | -1.82 percentage points |
| Ticket Status | 8,469 | 34.02% | 33.85% | -0.17 percentage points |

Neither text baseline beats the majority baseline in this signal check. This
does **not** establish that the labels were randomly generated; it means this
text baseline did not improve accuracy under this evaluation.

| Dataset audit | Result |
|---|---:|
| Rows / expected columns | 8,469 / 17 |
| Repeated rows by normalized ticket text | 144 |
| Exact-text groups with conflicting type, priority, or status | 104 |
| Near-duplicate pairs at TF-IDF cosine similarity >= 0.92 | 6 of 400 sampled distinct texts |

The near-duplicate scan is a deterministic sample, not a census. Of the six
sampled pairs, five had disjoint priority labels and two had disjoint status
labels. These conflicts and baseline results are data-quality warnings, not
proof that labels are incorrect. Recompute the audit after changing the
dataset; these figures are not future or production performance estimates.

## Research and Reproducibility

Interactive model reports use repeated stratified, exact-text-grouped
cross-validation across multiple seeds, descriptive confidence intervals,
and a separate holdout when class support permits. Chronological comparisons
are reported only when timestamps are explicitly trusted and there is enough
data. Final fitted models are used for prediction; they are separate from
evaluation estimates.

The Research baseline comparison includes majority class, Complement Naive
Bayes, linear SVM, TF-IDF + Logistic Regression, and an embedding +
Logistic Regression path. Sentence-transformer embeddings run only when the
optional package and `OPSLYRA_EMBEDDING_MODEL` are configured; otherwise the
embedding path uses TF-IDF + TruncatedSVD.

### Reproducible CLI experiments

The experiment runner can write dataset audit, model evaluation,
calibration/ablation, open-set, and risk–coverage JSON records under
`instance/experiments/`.

```powershell
python experiments.py --output instance/experiments --seeds 0,1,2 --folds 3 --max-rows 800 reproduce
```

Individual commands include `audit`, `evaluate`, `ablation`, `openset`,
`risk-coverage`, and `kappa`. Options go before the subcommand. For example:

```powershell
python experiments.py --output instance/experiments --seeds 0,1,2 --unknown "Refund request" openset
```

If GNU Make is installed, run the standard suite with `make reproduce`.
Independent label agreement is not currently measured. To calculate Cohen's
kappa from two independently produced annotation CSVs, see the
[annotation guide](docs/ANNOTATION_GUIDE.md). The bundled labels are not
independently verified operational ground truth.

## Installation and Run

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:OPSLYRA_SECRET_KEY = "use-a-long-random-secret"
python app.py
```

Open <http://127.0.0.1:5000>.

### macOS or Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPSLYRA_SECRET_KEY="use-a-long-random-secret"
python app.py
```

Use `requirements-lock.txt` for the pinned dependency set or `requirements.txt`
for the supported compatibility range.

### Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TICKETMIND_DATASET` | Input CSV path | `customer_support_tickets.csv` beside `app.py` |
| `OPSLYRA_SECRET_KEY` | Flask session signing key | Random per process if unset |
| `HOST` | Development server bind address | `127.0.0.1` |
| `PORT` | Development server port | `5000` |
| `FLASK_DEBUG` | Enable Flask debug mode when set to `true` | Disabled |

To use a different dataset:

```powershell
$env:TICKETMIND_DATASET = "C:\path\to\tickets.csv"
python app.py
```

Include all required columns. For an alternate CSV, a neighboring
`<csv-filename>.provenance.json` file can document source, citation, license,
retrieval date, and label/timestamp verification:

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

Only mark labels or timestamps verified when supporting evidence has been
independently checked and documented.

## Tests and CI

Run the test suite locally:

```powershell
python -m pytest -q
```

GitHub Actions runs pytest on Python 3.10, 3.11, and 3.13. Python 3.10 and
3.11 use `requirements.txt`; Python 3.13 uses the exact pins in
`requirements-lock.txt`. The workflow is `.github/workflows/ci.yml`.

## Technology

- Python and Flask
- SQLite for local state
- pandas, NumPy, SciPy, and scikit-learn
- HTML, CSS, JavaScript, and inline SVG/chart visualizations
- pytest and GitHub Actions

## Important Limitations

- **Not a hosted SaaS service yet:** there is no authentication,
  authorization, organization management, tenant isolation, or hosted
  deployment configuration.
- **No live ticketing integration:** Opslyra does not connect to an external
  service desk, change source tickets, or send customer replies.
- **Dataset provenance:** the bundled dataset is identified by the project as
  Kaggle-sourced, but its exact listing, license, retrieval date, and label
  verification have not been independently recorded. The repository's MIT
  license does not license the dataset; check the original terms before reuse.
- **Model confidence:** confidence scores are not automatically calibrated.
  Validation risk–coverage points do not guarantee future risk.
- **Open-set evaluation:** held-out ticket classes are proxies for unseen
  incidents, not verified production out-of-distribution events.
- **Routing and SLA:** there are no assigned-team ground-truth labels or
  verified SLA-breach labels in the dataset; related outcomes are simulations.
- **Time-series forecasts:** available response-time dates are insufficient
  for reliable recurrence forecasting.
- **Privacy:** redaction is pattern-based and does not guarantee removal of all
  personal or sensitive information. Do not use real sensitive data in this
  local prototype without appropriate safeguards.
- **Production readiness:** a hosted product would additionally need
  authentication and tenant-scoped authorization, secure secret management,
  HTTPS, production serving, durable managed storage, migrations, backup and
  retention policies, operational monitoring, and security review.

## Detailed Product Guide

This section describes the current behavior of the application in more detail.
It is a user guide to the repository as it exists, not a promise of hosted
SaaS functionality.

### Start Here: A Typical Analyst Session

1. Start the Flask application locally.
2. Open the Overview page.
3. Review the current dataset indicators and the high-priority queue.
4. Open Tickets to search or filter individual records.
5. Inspect a ticket detail page before making any decision.
6. Use AI prediction to explore a separate subject/description example.
7. Review model evaluation and Graphs before interpreting a prediction.
8. Use the Incident Lab for similarity, novelty, or workflow simulations.
9. Use Adaptive AI only after reviewing its mode and policy settings.
10. Run Research experiments when a repeatable, more detailed analysis is
    needed.
11. Review Data quality if the dataset or its provenance has changed.
12. Run the tests after code or configuration changes.

### Analyst Roles in the Demonstration

The application does not implement authentication or role-based access.
The role descriptions below are suggested ways to explore the interface.

#### Ticket analyst

- Starts from the Overview triage panel.
- Uses the Ticket Workspace filters.
- Checks the source ticket and existing resolution.
- Uses AI prediction as a second opinion.
- Does not treat model output as verified ticket truth.

#### Model evaluator

- Reviews the evaluation protocol before comparing scores.
- Checks class-level support and confusion matrices.
- Compares the Graphs page against the fold-local majority baseline.
- Reads the limitations attached to calibration and open-set results.
- Uses Research when testing alternative methods or split settings.

#### Human reviewer

- Selects Copilot mode for recommendations that require approval.
- Checks the prediction labels and similar resolved examples.
- Approves, edits, or rejects a recommendation.
- Verifies the label overlay on the ticket detail page.
- Uses ticket-specific takeover to pause assistance.

#### Dataset steward

- Checks the dataset profile and provenance sidecar.
- Reviews missing values and label distributions.
- Runs the dataset audit before modeling.
- Keeps an unchanged source copy of the input CSV.
- Documents source, license, retrieval date, and validation status.

### Operations Dashboard

The Overview page is intended to answer a few basic operational questions
without implying a connection to a live service desk.

#### Queue summary

- Total open tickets are counted from the `Ticket Status` field.
- Priority review counts are based on tickets marked Open and either High or
  Critical.
- Waiting counts are based on the Pending Customer Response status.
- The displayed counts describe the loaded dataset snapshot.
- They do not update from an external helpdesk.
- They do not measure elapsed SLA time.
- They do not replace an internal escalation policy.

#### Priority review panel

- Highlights open High and Critical work.
- Sorts the visible work using recorded priority.
- Links each row to the ticket detail page.
- Displays the recorded ticket category and channel where available.
- Shows a next-step hint derived from existing record fields.
- The hint is guidance and is not an executed workflow.

#### Issue category distribution

- Counts observed `Ticket Type` labels.
- Calculates shares from labeled records.
- Excludes missing type labels from the labeled denominator.
- Links users to the ticket explorer and incident graph workflows.
- Does not infer future demand.

#### Priority mix

- Summarizes observed `Ticket Priority` labels.
- Displays counts and proportions for the available labels.
- Does not adjust or normalize the source labels.

#### Status breakdown

- Summarizes observed `Ticket Status` values.
- Uses the dataset's current values rather than a live queue.
- Does not infer ticket lifecycle transitions.

#### Satisfaction summary

- Calculates the mean from available numeric satisfaction ratings.
- Missing ratings are excluded from the calculation.
- The average is not an independently collected survey metric.
- Ratings with invalid values are surfaced by Data quality checks.

### Ticket Workspace

The Ticket Workspace is the CSV-backed ticket browser.

#### Search

- Search is available across ticket ID, subject, and description.
- Search is case-insensitive.
- Search text is interpreted literally, not as a regular expression.
- Empty or whitespace-only search values do not restrict the results.
- Displayed subject and description text are passed through pattern-based
  redaction.

#### Filters

- Ticket type uses values observed in the dataset.
- Priority uses values observed in the dataset.
- Status uses values observed in the dataset.
- Multiple selected filters are combined.
- Filter comparison is case-insensitive.
- The clear action returns the user to the unfiltered ticket list.

#### Pagination

- The page size is ten records.
- Page numbers are clamped to the available range.
- Search and filter selections are retained when moving between pages.
- Pagination is computed from the filtered result count.

#### Ticket table

- Displays ticket details, type, priority, and status.
- Ticket subject links open the detail page.
- Ticket ID is shown for record identification.
- Personal names and email addresses are not displayed in the table.
- Channel and product are intentionally not shown as table columns.
- Empty-result states provide a way to clear filters.

#### Ticket detail

- Shows the ticket subject and record identifier.
- Shows the ticket's existing classification labels and status.
- Presents the description and existing resolution as separate sections.
- Presents attributes in a card-based layout.
- Redacts common identifiers from subject, description, and resolution text.
- Shows a human takeover notice when takeover is the effective ticket mode.
- Shows saved category and priority overlays when local approved labels exist.
- Shows AI activity events associated with the ticket.
- Similar-case links lead to other ticket detail pages.
- The source CSV remains unchanged when an overlay is saved.

### AI Prediction

The prediction page accepts a subject and description as independent optional
text fields.

#### Prediction input

- Subject input is limited to 500 characters.
- Description input is limited to 10,000 characters.
- Empty text can result in insufficient evidence or model-specific behavior.
- Email, phone-like numbers, IP addresses, usernames, and credential-like
  patterns are redacted before supported model paths.
- The raw CSV is not changed by an interactive prediction.

#### Prediction output

- Ticket type and ticket priority use separate classifiers.
- Class probabilities appear only when the estimator supports them.
- Confidence values are uncalibrated model outputs unless separately evaluated.
- The displayed top terms are model-specific feature contributions.
- Feature contributions are not causal explanations.
- A configured open-set threshold may produce an UNKNOWN/abstain result.
- When abstaining, the page may also show the closed-set class estimate for
  transparency.
- The page tells users to review predictions before acting.

#### Use predictions responsibly

- Confirm the underlying ticket text.
- Check the model evaluation report.
- Consider label support and class-level performance.
- Treat unfamiliar or high-risk content as requiring human review.
- Do not infer probability calibration from a confidence percentage.
- Do not treat similar tickets as guaranteed resolutions.

### Model Evaluation

The Model evaluation page reports baseline performance for available targets.

#### Training action

- The page can train or refresh baseline models.
- Training can create a model artifact and a local model version record.
- A failed training attempt is surfaced as an application message.
- A report from an older format is identified as outdated where detectable.
- Training on the full eligible dataset is distinct from cross-validation.

#### Evaluation summaries

- Reports include model accuracy where available.
- Reports include macro and weighted F1 where available.
- Descriptive confidence intervals are displayed for supported metrics.
- Results are grouped by prediction target.
- Per-class cards show F1, precision, recall, and support.
- Per-class tables provide the numeric report values.
- A repeated grouped-CV metadata row describes the evaluation runs.

#### Interpreting scores

- Accuracy is the share of correctly predicted examples.
- Macro F1 gives each class equal weight.
- Weighted F1 weights classes by observed support.
- Class support is the number of evaluated examples for that class.
- A high overall score can hide weak minority-class performance.
- Confidence intervals describe observed fold/seed variation in this setup.
- They are not guarantees of future model performance.

#### Training and model lifecycle

- Baseline estimators are created from the current configured dataset.
- Dataset fingerprints help detect changes to the underlying data.
- Model artifacts are stored beneath the configured artifact directory.
- Model versions are recorded in the local SQLite store.
- The Incident Lab can train a new version after reviewed corrections exist.
- A previous saved model version can be restored using rollback.
- A rollback selects a stored artifact; it does not revert the CSV.
- Artifacts are local and are not shared across user accounts.

### Graphs

The Graphs page combines label-signal checks and probability calibration
diagnostics.

#### Cross-validated accuracy versus majority baseline

- Uses ticket subject and description as the text input.
- Evaluates Ticket Priority and Ticket Status for the bundled signal check.
- Uses three-fold stratified grouped cross-validation.
- Keeps normalized exact-text groups in one fold.
- Fits the text model and majority baseline on each training fold only.
- Reports labeled rows and distinct text groups.
- Shows mean accuracy and fold-level results.
- Calculates the difference in percentage points.
- Exact-text grouping does not prevent nonidentical near-duplicates from
  crossing folds.
- A result below the majority baseline is not proof of random labels.

#### Duplicate-text audit

- Groups normalized exact text.
- Checks whether duplicate text groups contain conflicting labels.
- Reports label conflicts by available target.
- Runs a deterministic sample-based near-duplicate analysis.
- Uses TF-IDF similarity to identify candidate near-duplicate pairs.
- The current near-duplicate audit is bounded to a sample of distinct text.
- The displayed near-duplicate count is not a census of every pair.
- Similarity is a text measure, not proof of duplicate operational incidents.

#### Probability calibration

- Supports Ticket Type or Ticket Priority as the selected target.
- Uses an untouched text-grouped holdout when a valid split can be made.
- Compares uncalibrated and calibration-method outputs where supported.
- Displays a reliability diagram.
- Reports expected calibration error by method.
- Includes a perfect-calibration reference line.
- Calibration results depend on the holdout size and class support.
- Calibration does not guarantee future confidence quality.

#### Graph downloads

- Supported graphs can be exported as PNG files.
- The download controls request 1200 DPI metadata.
- Resolution metadata does not create additional measured information.
- Exported charts remain visual representations of the current analysis.

### Data Quality

The Data quality page inspects the loaded dataset without rewriting its values.

#### Schema checks

- Checks required fields.
- Displays observed data types.
- Reports duplicate ticket IDs.
- Reports missing ticket IDs.
- Checks parseability of numeric fields.
- Checks customer satisfaction rating values against the expected 1–5 range.
- Checks parseability of date/time columns.

#### Missing values

- Counts blank strings and null values as missing after ingestion normalization.
- Shows field-level missing counts and percentages.
- Does not impute missing labels or outcomes.
- Does not create synthetic resolutions or satisfaction scores.
- Missing values remain missing in analysis copies.

#### Label inspection

- Shows the observed distribution of Ticket Type.
- Shows the observed distribution of Ticket Priority.
- Shows the observed distribution of Ticket Status.
- Shows the observed distribution of Ticket Channel.
- Reports distinct label counts.
- Reports case or whitespace variants that may indicate inconsistent spelling.
- Label distributions describe the file, not confirmed ground truth.

#### Timestamp inspection

- Reports present, parseable, and unparseable values for date fields.
- Does not infer durations from timestamps.
- Does not assert chronology is trustworthy unless provenance says so.
- Does not claim recurrence forecasts when data support is inadequate.

### Adaptive AI: Detailed Mode Behavior

Adaptive AI settings are stored in the local workspace state.

#### Workspace mode

- Workspace mode is the default for tickets without a ticket-specific override.
- A workspace mode change records a workspace activity event.
- Available workspace modes are Manual, Copilot, Selective Automation, and
  Autonomous Workflow.
- Human Takeover is a ticket-specific mode, not a workspace mode.
- An invalid mode request is rejected.

#### Ticket mode

- A ticket can inherit the workspace mode.
- A ticket can use an explicit ticket-specific mode.
- Selecting workspace mode clears the ticket-specific override.
- Selecting takeover pauses AI processing for that ticket.
- Ticket mode changes are written to the activity timeline.
- Ticket mode does not update the source dataset.

#### Policy actions

- The policy can enable the category action.
- The policy can enable the priority action.
- Other action names are rejected by the policy endpoint.
- The target-coverage value is bounded between 0.10 and 1.00.
- The target-selective-risk value is bounded between 0.00 and 1.00.
- Selection mode is either target coverage or target selective risk.
- Per-target cutoffs are selected from validation risk–coverage information.
- A policy update is saved locally and added to workspace activity.

#### Copilot recommendations

- Copilot prepares category and priority predictions.
- A recommendation is saved as pending approval.
- A human may approve, edit, or reject it.
- Approval validates proposed labels against labels observed in the dataset.
- Approval saves a local overlay.
- Rejection retains the decision in activity history.
- Repeated requests can use idempotency keys to avoid duplicate actions.
- No external ticket update is sent.

#### Selective and autonomous workflows

- Only explicitly enabled category and priority actions can be considered.
- A target-specific validation cutoff is checked for each action.
- Sensitive wording or a Critical prediction can trigger review flags.
- Scores below a selected cutoff can stop the workflow for review.
- Enabled predictions can be written only to the local label overlay.
- Suggested routing is informational and is not executed.
- Customer messages are not generated or sent.
- External integrations are not invoked.
- The name Autonomous does not imply autonomous operation of a real service
  desk.

#### Dry-run and policy preview

- Dry-run endpoints return a preview-oriented workflow evaluation.
- Dry-run is designed not to commit ticket overlay or workflow side effects.
- Policy preview can be used to inspect the outcome of a proposed policy.
- A preview is temporary session state.
- Recommendations check preview identity and context before use.
- Previews expire after a bounded interval.
- A changed policy or dataset invalidates stale preview context.

### Incident Lab: Detailed Workflow Notes

#### 1. Human-in-the-loop learning

- Accepts a ticket identifier.
- Supports Ticket Type, Ticket Priority, and Team correction targets.
- Requires a nonempty corrected label.
- Stores the original label when available.
- Estimates informativeness from model uncertainty where a model exists.
- Saves reviewed examples in local SQLite state.
- Ranks visible corrections by informativeness and time.
- Training can create a new model version using corrections.
- Corrections do not change the source CSV.
- Team labels do not imply a verified team-routing ground truth.

#### 2. Open-set incident discovery

- Accepts a subject, description, or both.
- Compares text with the observed ticket corpus.
- Shows nearest similarity and local-density signals.
- Can display nearest example tickets.
- Uses a validation-derived reference threshold when available.
- Uses threshold and density signals to distinguish known-like from unfamiliar.
- The result is a novelty diagnostic, not a production OOD guarantee.
- The nearest tickets are evidence for review, not recommended fixes.

#### 3. Counterfactual text explorer

- Accepts a subject and description.
- Creates alternate wording scenarios.
- Runs the model against the original and modified text.
- Shows labels and scores for each scenario.
- Helps explore sensitivity to context changes.
- Does not estimate causal effects.
- Does not prove a scenario will occur in production.
- Does not update the ticket record.

#### 4. Incident relationship graph

- Accepts a center ticket identifier.
- Ranks nearby records using text and available context.
- Displays the center ticket separately from related ticket cards.
- Shows relationship scores and reasons where available.
- Can use shared product, error-code, and time information.
- Links related cards to their ticket detail pages.
- Similarity does not prove common root cause.
- Graph edges are analytical suggestions, not incident-management actions.

#### 5. SLA risk sandbox

- Accepts priority, elapsed hours, target hours, and queue load.
- Computes a transparent risk index from those inputs.
- Exposes assumptions as interactive controls.
- Does not read an operational SLA contract.
- Does not know actual staffing or service-level commitments.
- Does not predict a verified SLA breach.
- Use it as a scenario illustration only.

#### 6. Adaptive routing simulator

- Compares a capability/skill-based suggestion with round-robin allocation.
- Uses configured capability maps and load assumptions.
- Can vary capacity factor within the accepted input range.
- Displays simulated outcomes and route details.
- Does not learn verified team preferences without reviewed team labels.
- Does not dispatch or reassign a real ticket.
- Team skill, capacity, and processing-time assumptions are not dataset truth.

#### 7. Recurrence and early-warning radar

- Summarizes available dated records and text clusters.
- Examines observed historical counts rather than a live event stream.
- Needs enough distinct trustworthy response dates for chronological checks.
- May withhold recurrence metrics when temporal support is insufficient.
- Does not claim seasonality from sparse or untrusted dates.
- Does not issue a real operational alert.
- Event detection precision and recall are unavailable without ground-truth
  event labels.

#### 8. Privacy-aware text handling

- Accepts text for a local redaction scan.
- Recognizes common email, username, IP, secret, bearer token, and phone-like
  patterns.
- Shows redacted text and per-pattern counts.
- Records count-oriented audit information locally.
- Does not claim complete anonymization.
- Avoid submitting real credentials or sensitive personal data to the
  prototype.

#### 9. Evidence-grounded resolution memory

- Accepts a subject and/or description.
- Searches historical tickets with recorded resolutions.
- Returns a bounded list of similar cases.
- Displays the source ticket identifiers and similarity.
- Requires a human to confirm that the resolution applies.
- Similar wording does not guarantee the same underlying cause.
- Does not automatically copy a resolution to another ticket.

#### 10. Model reliability observatory

- Shows saved model versions.
- Displays available metrics and training-example counts.
- Provides rollback to an existing local model artifact.
- Uses a chronological comparison only when sufficient trusted date data exist.
- Shows unavailable results as unavailable rather than fabricating a metric.
- Does not guarantee an artifact is appropriate for a changed dataset.

### Research Modules: Detailed Notes

#### Open-set experiment

- Selects Ticket Type or Ticket Priority as the target.
- Allows unknown class labels to be selected or selected by a rare-class rule.
- Accepts 1–20 unique random seeds.
- Uses a bounded sample for responsive interaction.
- Keeps exact normalized text groups together in splitting.
- Reports methods and metrics per valid seed.
- Selects unknown rejection thresholds from validation known-class scores.
- Reports AUROC and AUPR when both evaluation classes are present.
- Reports unknown recall and known rejection rate where defined.
- Reports known-class classification performance separately.
- Reports skipped seeds when splits cannot meet the experiment requirements.
- A held-out known label is a proxy for an unseen incident class.
- Results do not prove deployment-level open-set detection quality.

#### Supporting model comparison

- Selects Ticket Type or Ticket Priority.
- Accepts unique random seeds.
- Uses three-fold grouped validation in the interactive form.
- Limits the number of rows for responsive interaction.
- Compares a majority baseline with supported text classifiers.
- Reports model performance using common classification metrics.
- Supports a text embedding path when the optional dependency and model
  configuration are available.
- Uses TF-IDF plus TruncatedSVD as a fallback embedding representation.
- Does not silently convert missing target labels into synthetic classes.

#### Dataset and label-signal audit

- Reports dataset-level properties and label quality diagnostics.
- Compares text classification to a majority baseline.
- Counts exact-text duplicate groups.
- Detects label conflicts within identical normalized text.
- Samples near-duplicate candidate pairs deterministically.
- Reviews potential leakage and unsuitable features.
- Provides dataset suitability diagnostics.
- Audit findings identify risks for review; they do not prove label intent.

#### Calibration and feature ablation

- Calibration evaluates confidence versus observed correctness.
- Reliability diagrams group predictions into confidence bins.
- Expected calibration error summarizes bin-level discrepancy.
- Calibration methods are assessed on a held-out split when feasible.
- Feature ablation compares configured input feature sets.
- Ablation scores are conditional on the split and sample.
- Smaller interactive sample caps improve responsiveness but increase
  uncertainty.

#### Risk–coverage research

- Evaluates selective prediction at varying score thresholds.
- Coverage is the share of cases accepted by the model.
- Selective risk is error among accepted cases.
- Target-coverage mode selects a threshold for requested coverage.
- Target-risk mode selects a threshold meeting a requested risk where possible.
- Thresholds are based on validation observations.
- Risk can be undefined when no examples are accepted.
- A target selected on validation does not guarantee future risk.

#### Active-learning queue

- Orders candidate tickets for human review.
- Uses uncertainty, novelty, impact proxy, and diversity components.
- Allows the queue weights to be adjusted within input bounds.
- Normalizes provided weights before ranking.
- Returns a bounded set of candidates.
- Annotation effort is estimated from text characteristics.
- Impact is a text-keyword proxy, not a measured business outcome.
- The queue prioritizes review; it does not label tickets automatically.

#### Active-learning benchmark

- Simulates label acquisition under a fixed evaluation setup.
- Compares selection policies at budget checkpoints.
- Uses the same seed labels and budget schedule across policies.
- Uses labeled ticket-type examples from the configured dataset.
- Reports metrics only when the split and class support permit.
- Does not model actual annotation labor or reviewer disagreement unless such
  labels are supplied.
- The benchmark is a simulation, not an observed staffing experiment.

#### Constrained routing research

- Accepts ticket text and a remaining SLA budget.
- Accepts team workload and capacity values.
- Compares model text recommendations with configured constraints.
- Presents a proposed route rather than executing an assignment.
- Shows skill/capacity/SLA feasibility under the submitted assumptions.
- Does not provide a learned routing outcome without suitable reviewed team
  labels.
- Does not measure real operational processing time.

#### Temporal fingerprint research

- Builds text and context features from eligible ticket fields.
- Can incorporate product identifiers, extracted error codes, and timestamps.
- Compares temporal-aware fingerprints with text-only clustering.
- Needs valid timestamps and sufficient chronological support.
- Uses chronological splits rather than random shuffling when suitable.
- May withhold recurrence and forecast-quality claims when evidence is weak.
- Does not equate clusters with verified incidents.

## HTTP Route Reference

This reference describes the routes implemented by the local Flask app.
It is not a public API contract or a promise of backward compatibility.

### Browser pages

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Render the Overview dashboard. |
| `GET` | `/tickets` | Search, filter, and paginate ticket records. |
| `GET` | `/tickets/<ticket_id>` | Render an individual ticket and local activity. |
| `GET` | `/predict` | Render the prediction form. |
| `POST` | `/predict` | Run local classification for submitted text. |
| `GET` | `/models` | Render model evaluation reports. |
| `POST` | `/models` | Train or refresh baseline models. |
| `GET` | `/graphs` | Render label signal and calibration charts. |
| `GET` | `/data-quality` | Render schema and completeness checks. |
| `GET` | `/intelligence` | Render the Incident Lab. |
| `POST` | `/intelligence` | Run the selected local Incident Lab action. |
| `GET` | `/adaptive-ai` | Render modes, policies, and risk–coverage diagnostics. |
| `GET` | `/research` | Render Research modules. |
| `POST` | `/research` | Run the selected research experiment. |

### Adaptive AI JSON endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/adaptive-ai/mode` | Set the workspace mode. |
| `POST` | `/api/adaptive-ai/policy` | Validate and save the workspace policy. |
| `POST` | `/api/tickets/<ticket_id>/adaptive-mode` | Set or clear a ticket mode override. |
| `POST` | `/api/tickets/<ticket_id>/ai/dry-run` | Preview a ticket workflow without intended persistence. |
| `POST` | `/api/tickets/<ticket_id>/ai/policy-preview` | Preview a policy for a ticket. |
| `POST` | `/api/tickets/<ticket_id>/ai/recommend` | Create or reuse a ticket recommendation. |
| `POST` | `/api/tickets/<ticket_id>/ai/recommendations/<activity_id>/<resolution>` | Approve or reject a pending recommendation. |

### JSON request conventions

- Requests use a JSON object for adaptive API operations.
- Invalid or missing JSON objects are rejected.
- Unsafe requests require a valid session-bound CSRF token.
- Send the CSRF token in the `X-CSRF-Token` header for JSON operations.
- An `Idempotency-Key` header can identify retryable actions.
- Ticket identifiers must refer to a record in the configured dataset.
- Unsupported mode names and action names are rejected.
- Label approval accepts only observed Ticket Type and Ticket Priority labels.
- Error responses use non-success HTTP statuses for rejected operations.

### Common response behavior

- `200` indicates a successful page or request.
- `400` indicates invalid input or unsupported values.
- `403` indicates a failed CSRF check.
- `404` indicates a missing ticket or resource.
- `409` indicates a workflow conflict, such as a paused mode or already
  resolved recommendation.
- `503` can indicate model preparation or training is unavailable.
- Response details can vary by route and are not versioned as a public SDK.

## Dataset Schema Reference

The loader expects the following 17 columns. Additional columns are ignored
after ingestion so feature use remains explicit.

| Field | Role in this application |
|---|---|
| `Ticket ID` | Record identifier used for lookup and display. |
| `Customer Name` | Personal identifier; excluded from model features and hidden in ticket lists. |
| `Customer Email` | Personal identifier; excluded from model features and hidden in ticket lists. |
| `Customer Age` | Data-quality numeric check; excluded from text-model features. |
| `Customer Gender` | Personal/demographic field; excluded from text-model features. |
| `Product Purchased` | Ticket attribute and permitted metadata for selected analysis. |
| `Date of Purchase` | Date-parseability check; not a model text feature. |
| `Ticket Type` | Classification target and observed category label. |
| `Ticket Subject` | Permitted text feature and ticket search field. |
| `Ticket Description` | Permitted text feature and ticket search field. |
| `Ticket Status` | Observed workflow label and signal-audit target; excluded from model features. |
| `Resolution` | Detail-page content and similar-resolution evidence; excluded from prediction features. |
| `Ticket Priority` | Classification target and observed priority label. |
| `Ticket Channel` | Ticket attribute and permitted metadata for selected analysis. |
| `First Response Time` | Date/time parseability and supported chronological analysis. |
| `Time to Resolution` | Date/time parseability and supported chronological analysis. |
| `Customer Satisfaction Rating` | Numeric range check and dashboard summary; excluded from text-model features. |

### Required input rules

- All 17 required column names must be present.
- The CSV must contain at least one row.
- `Ticket ID` is read as text to preserve identifier formatting.
- Blank-only strings are normalized to missing values.
- The loader does not invent labels.
- The loader does not impute numeric values.
- The source frame is retained as raw input for reporting.
- A dataset fingerprint is attached to the loaded frame.
- Dataset provenance is loaded from a neighboring JSON sidecar when present.

### Prediction feature boundary

- Ticket subject is a permitted text feature.
- Ticket description is a permitted text feature.
- Product and channel are permitted metadata for selected analyses.
- Customer name is excluded.
- Customer email is excluded.
- Customer age is excluded.
- Customer gender is excluded.
- Ticket ID is excluded.
- Ticket status is excluded from prediction inputs.
- Resolution is excluded.
- Time to resolution is excluded.
- Customer satisfaction is excluded.
- First response time is excluded.

### Provenance sidecar

The sidecar file is named by adding `.provenance.json` after the CSV suffix.
For example, `support.csv` uses `support.csv.provenance.json`.

The sidecar may document:

- The source kind.
- The source name or URL.
- A human-readable dataset label.
- A citation.
- A license identifier.
- The retrieval date.
- Whether labels have been verified.
- Whether timestamps are trustworthy.

The current loader validates that `labels_verified` and
`timestamps_trustworthy`, when supplied, are boolean values.
Only set these fields to true when the supporting evidence has been checked.

## Modeling and Evaluation Reference

### Text preparation

- Missing subject values become empty strings for vectorization.
- Missing description values become empty strings for vectorization.
- Supported model text paths redact common sensitive patterns.
- Subject and description are combined for the standard ticket classifier.
- The feature boundary is controlled by the data preparation utilities.
- The text vectorizer learns vocabulary only from the training partition.
- Evaluation code should fit preprocessing inside each training fold.

### Baseline classifier

- The baseline ticket classifier uses TF-IDF text features.
- Ticket Type and Ticket Priority are trained as separate estimators.
- Logistic Regression provides the primary interactive prediction baseline.
- The prediction output can include per-class probabilities when supported.
- The displayed scores are not automatically calibrated.
- Top-term explanations are based on the fitted model's feature weights.

### Research model zoo

The research comparison includes these model paths:

- Most-frequent majority classifier.
- Complement Naive Bayes with TF-IDF features.
- Linear support vector machine with TF-IDF features.
- Logistic Regression with TF-IDF word n-grams.
- Embedding plus Logistic Regression.
- TF-IDF plus TruncatedSVD as an embedding-path fallback.

The optional sentence-transformer path requires its package and configured
model to be available. The project does not download a model automatically as
part of its default local setup.

### Grouped split strategy

- Text is case-folded and normalized for whitespace before equality grouping.
- Exact normalized text rows share an integer group ID.
- Group IDs are kept together across train and test splits.
- Stratified group splitting is attempted first.
- A bounded group assignment search can be used for small datasets when the
  first splitter cannot produce class-complete partitions.
- The fallback is bounded by the number of unique text groups.
- Splits require class support in both train and test where the caller needs
  class-complete evaluation.
- Exact-text grouping does not eliminate all leakage from near-duplicate
  wording.

### Metrics

#### Accuracy

Accuracy is the number of correct predictions divided by the number of
evaluated predictions.

Accuracy is easy to interpret but can obscure weak performance on rare
classes.

Always read it alongside class support and per-class scores.

#### Precision

Precision measures the share of predicted examples of a class that are
correct.

Low precision means more false-positive assignments for that class.

Precision should be interpreted with the target class support and confusion
matrix.

#### Recall

Recall measures the share of true examples of a class that are recovered.

Low recall means more examples of that class were missed.

Recall is not an event-detection metric unless event labels are available.

#### F1

F1 is the harmonic mean of precision and recall.

Macro F1 computes F1 per class and averages classes equally.

Weighted F1 weights class F1 by class support.

The chosen average changes how class imbalance affects the result.

#### AUROC

AUROC measures ranking discrimination across thresholds for a binary scoring
task.

For open-set analysis, it measures the ability to rank unknown examples above
known examples under the selected score convention.

It does not specify a deployable threshold on its own.

#### AUPR

AUPR summarizes the precision–recall relationship for the positive class.

It depends on class prevalence and the positive-class definition.

Do not compare AUPR values without checking those conditions.

#### Expected calibration error

Expected calibration error summarizes discrepancies between confidence and
empirical correctness across bins.

It depends on binning and sample count.

It is not a complete calibration diagnosis by itself.

#### Risk and coverage

Coverage is the portion of examples accepted by a selective classifier.

Selective risk is the error rate among accepted examples.

Increasing a score threshold often reduces coverage while changing risk.

The observed validation curve is evidence about that validation sample only.

### Intervals and seeds

- Multiple seeds can reveal variability from randomized splits.
- Fold-level results are retained where appropriate.
- Confidence intervals are descriptive summaries of observed scores.
- They are not a guarantee of statistical coverage under every assumption.
- A small number of folds or seeds limits precision.
- Unsupported splits are skipped or marked unavailable rather than fabricated.
- Users should inspect sample counts for each metric.

### Dataset result tables

The README intentionally contains two dataset-results tables:

1. The model signal table compares the text model with the majority baseline.
2. The dataset audit table summarizes size, repeated text, label conflicts,
   and sampled near-duplicate pairs.

These are snapshot results from the bundled CSV.

They must be recomputed after changing the data or evaluation implementation.

They must not be presented as service-wide production metrics.

## Local Storage and State

### Storage locations

- `instance/ticketmind.sqlite3` stores application state by default.
- `instance/models/` stores serialized model artifacts by default.
- `instance/experiments/` stores experiment snapshots by default.
- These paths are created as needed by the application.
- `instance/` is ignored by Git.
- The configured input CSV remains separate from state storage.

### State categories

Local SQLite state can include:

- Reviewed correction rows.
- Saved model version metadata.
- Redaction count audit entries.
- Novelty event summaries.
- Workspace Adaptive AI settings.
- Ticket-specific mode overrides.
- Ticket label overlays.
- AI workflow activity.

### Label overlays

- Overlays store a ticket ID and optional ticket type and priority values.
- Approvals require labels present in the loaded dataset.
- An overlay does not rewrite the CSV.
- Overlays are local to the current database file.
- The detail page shows overlay labels when present.
- Overlay data is not synchronized to a central service.
- Removing or replacing the local database removes its stored overlay state.

### Model artifacts

- Model files are stored under the artifact directory.
- The database records artifact paths and version metrics.
- The active version is marked in local state.
- Rollback marks a saved version active and loads its artifact.
- Keep artifact files and the database together for a consistent local history.
- Do not load untrusted serialized model artifacts.
- Artifact backups are not configured automatically.

### Experiment records

- CLI experiments write JSON records to the selected output directory.
- The default output directory is `instance/experiments`.
- Records can include configuration, metrics, dataset fingerprint, and
  software versions.
- Experiment outputs may contain local paths if path metadata is recorded.
- Review outputs before sharing them externally.

### Resetting a local workspace

The project does not currently provide an in-app data-retention wizard.

To reset local demo state:

1. Stop the application.
2. Back up any database, models, or experiment results that must be retained.
3. Remove only the specific local state files or directories you intend to
   reset.
4. Restart the application to initialize fresh local state.
5. Re-run any required model training or experiments.

Do not remove the source dataset when intending only to clear local state.

Do not commit local databases, model artifacts, or experiment outputs unless
they are intentionally prepared for sharing.

## Privacy and Security Notes

### Pattern-based redaction

The redaction helper searches for common patterns such as:

- Email addresses.
- Username-like `@name` strings.
- IPv4 addresses.
- Credential assignments such as `api_key=...`.
- Bearer-token-like strings.
- Username or login assignments.
- Phone-like number sequences.

Redaction is a best-effort pattern scan.

It can miss personal data that does not match these patterns.

It can redact benign strings that resemble identifiers.

It does not replace a privacy review or anonymization process.

### Data minimization in model features

- Customer names are not model features.
- Customer emails are not model features.
- Demographic fields are not model features.
- Ticket IDs are not model features.
- Status, resolution, satisfaction, and time outcomes are excluded from the
  standard ticket text model.
- Text inputs are redacted before supported prediction and analysis paths.
- Review the explicit feature allowlist before adding new features.

### CSRF protection

- Unsafe HTTP methods are checked by a global request hook.
- HTML forms include a session-bound CSRF token.
- JSON requests send the token through `X-CSRF-Token`.
- Requests with a missing or invalid token are rejected.
- CSRF protection does not authenticate a user.
- CSRF protection does not authorize access to a tenant.

### Local development secret

- Set `OPSLYRA_SECRET_KEY` to a long random value for a stable local session
  signing key.
- When unset, the application generates a per-process random key.
- A per-process key invalidates sessions on restart.
- Do not put secrets in source code or commit them.
- Do not reuse a development key in a hosted deployment.

### Safe use of the demo

- Use synthetic or approved test data for exploration.
- Do not submit production credentials to any text field.
- Do not assume a local workstation is an approved data environment.
- Restrict filesystem access to local state and uploaded datasets.
- Avoid sharing diagnostic output containing paths or raw text.
- Review model artifacts and experiment JSON before publishing.
- Verify the original dataset license before redistribution.

## Troubleshooting

### The app cannot find the dataset

1. Confirm `customer_support_tickets.csv` exists beside `app.py`, or set
   `TICKETMIND_DATASET`.
2. Confirm the configured path points to a readable CSV file.
3. Confirm the file uses the required column names.
4. Confirm the file contains at least one data row.
5. Read the visible dataset error message before changing code.

Example on Windows PowerShell:

```powershell
$env:TICKETMIND_DATASET = "C:\data\support.csv"
python app.py
```

### Required columns are missing

- Compare the CSV header to the [Dataset Schema Reference](#dataset-schema-reference).
- Preserve the expected spelling and spacing of column names.
- Do not remove columns simply because a screen does not display them.
- Keep the required schema even when values are missing.
- Avoid fabricating values to satisfy the loader.

### A prediction is missing or abstains

- Confirm that subject or description text was submitted.
- Check whether the model has been trained for the current dataset.
- Review the evaluation report for class support.
- Check whether an open-set threshold is enabled.
- An UNKNOWN/abstain decision is a supported outcome.
- Do not force a closed-set label when the evidence is insufficient.

### Model evaluation is outdated

- Use the Train baseline action on the Model evaluation page.
- Confirm that model training completes without an error message.
- Check whether the active artifact matches the current dataset fingerprint.
- Do not manually edit a serialized model file.
- If an artifact is missing, retrain rather than assuming old metrics apply.

### Open-set experiment reports skipped seeds

- Inspect the selected target's label counts.
- Confirm each selected class has enough distinct text groups.
- Try a different unknown class only when it is a valid research choice.
- Increase the number of seeds within the supported limit.
- Do not treat skipped seeds as failed predictions.
- Report the number of valid seeds alongside the metrics.

### Calibration results are unavailable

- Confirm that the target has sufficient labeled class support.
- Check whether grouped splitting can place all relevant labels in the
  required partitions.
- Review dataset duplicates and class imbalance.
- Do not remove grouping solely to force a calibration result.
- Record unavailable calibration as a limitation.

### Temporal analysis is withheld

- Review timestamp parseability in Data quality.
- Check the provenance sidecar's `timestamps_trustworthy` value.
- Confirm there are enough distinct chronological dates.
- Do not invent dates or mark timestamps trustworthy without evidence.
- A withheld temporal result is preferable to a misleading forecast.

### A label overlay does not appear

- Confirm that the recommendation was approved rather than merely previewed.
- Open the same ticket ID that received the approval.
- Check the local SQLite file configured for the current run.
- Confirm the approval label was an observed dataset label.
- An overlay is local and is not written into the CSV.

### A ticket is paused in takeover

- Check the effective ticket mode in Adaptive AI.
- Select workspace mode for that ticket to clear a ticket-level override.
- Verify that the activity timeline records the mode change.
- A workspace mode change does not necessarily erase ticket-level mode
  overrides.

### CSRF request returns 403

- Load or refresh a page to obtain a current session token.
- Include the token in an HTML form field or JSON `X-CSRF-Token` header.
- Keep browser cookies enabled for the local session.
- Avoid replaying an expired browser session.

### Changes do not appear in the browser

- Confirm the Flask process is running the current checkout.
- Restart the development server after Python changes.
- Refresh the page after template and CSS edits.
- Clear stale browser cache if static assets appear unchanged.
- Check the terminal for template or server errors.

### Test suite fails locally

- Activate the intended virtual environment.
- Install the appropriate requirements file.
- Run the failing test in isolation to capture a focused traceback.
- Check whether the expected dataset file is available.
- Compare the selected Python version with the CI matrix.
- Run the full suite after resolving the focused failure.

## Development Guide

### Repository map

| Path | Responsibility |
|---|---|
| `app.py` | Flask application factory, HTTP routes, request validation, and orchestration. |
| `data.py` | CSV loading, schema validation, provenance, redaction, and dataset profile. |
| `dashboard.py` | Overview metrics and label distribution summaries. |
| `models.py` | Baseline training, metrics, model versions, prediction, and term contributions. |
| `evaluation.py` | Cross-validation, model comparison, calibration, ablation, and temporal evaluation helpers. |
| `dataset_audit.py` | Label-signal, duplicate, leakage, and dataset suitability diagnostics. |
| `intelligence.py` | Similar tickets, novelty, relationship graphs, recurrence, routing, SLA sandbox, and drift. |
| `adaptive.py` | Adaptive AI mode behavior and local workflow decision logic. |
| `risk_coverage.py` | Risk–coverage curves, threshold selection, and reviewer-feedback summaries. |
| `research.py` | Active learning, constrained routing, and temporal fingerprint experiments. |
| `openset.py` | Open-set experiment, scoring, thresholding, and per-seed metrics. |
| `reproducibility.py` | Text grouping, split helpers, fingerprints, metadata, intervals, and JSON output. |
| `state.py` | SQLite schema and state-store operations. |
| `experiments.py` | Command-line interface for repeatable research runs. |
| `templates/` | Jinja templates for workspace pages. |
| `static/` | CSS, JavaScript, and browser assets. |
| `tests/` | Unit and application tests. |
| `docs/ANNOTATION_GUIDE.md` | Independent annotation and agreement guidance. |
| `instance/` | Local database, models, and experiment outputs; ignored by Git. |

### Development setup

1. Create a virtual environment.
2. Activate it.
3. Install project dependencies.
4. Set a local secret key when stable sessions are useful.
5. Run the Flask application.
6. Run pytest before submitting a change.

Windows setup:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

### Change guidance

- Keep changes focused on the user-visible requirement.
- Preserve the local-only deployment boundary.
- Do not claim external integration without implementing and testing it.
- Keep the source dataset immutable during UI workflows.
- Add tests for changed behavior.
- Use grouped splitting when exact-text leakage matters.
- Preserve missing values unless a method explicitly handles them.
- Surface validation errors to the user.
- Keep unsupported metrics unavailable instead of using fabricated defaults.
- Update this README when application behavior changes.
- Avoid storing sensitive datasets or outputs in Git.

### Adding a new page

1. Define the page's role in the workspace.
2. Add a Flask route with explicit input validation.
3. Add a Jinja template extending `base.html`.
4. Add an accessible navigation link only when it is a top-level workspace
   area.
5. Add responsive styles in the established stylesheet.
6. Add tests for successful and invalid requests.
7. Document inputs, outputs, storage, and limitations.
8. Verify that unsafe form submissions include CSRF protection.

### Adding a model feature

1. Define the target and permitted feature inputs.
2. Verify that target leakage is not introduced.
3. Ensure preprocessing is fitted only on training data.
4. Keep duplicate text groups together when splitting.
5. Define behavior for missing labels and insufficient class support.
6. Add a baseline comparison.
7. Report per-class support and metrics.
8. Avoid presenting uncalibrated scores as probabilities.
9. Add tests for split isolation and result shape.
10. Document the feature boundary and limitations.

### Adding an experiment

1. Define the research question.
2. Define inclusion and exclusion rules.
3. Define primary and secondary metrics.
4. Define group and temporal split constraints.
5. Record random seeds and package versions.
6. Record dataset fingerprint and provenance.
7. Explain unsupported or skipped runs.
8. Add a small deterministic regression test.
9. Keep interactive sample caps visible.
10. Avoid causal language unless the study design supports it.

### Testing conventions

- Tests use pytest.
- Application tests use a Flask test client.
- Research tests use small deterministic datasets.
- Tests should not require external service credentials.
- Tests should not depend on a live helpdesk.
- Keep test artifacts in temporary locations where practical.
- Run focused tests during iteration.
- Run the full suite before finalizing.

### Continuous integration

- GitHub Actions runs on pushes and pull requests.
- CI currently tests Python 3.10, 3.11, and 3.13.
- Python 3.10 and 3.11 install the compatibility requirements.
- Python 3.13 installs the pinned lock file.
- CI executes `python -m pytest -q`.
- CI permissions are limited to repository contents read access.

## SaaS Productization Roadmap

The current UI is SaaS-style, but a production SaaS deployment requires
additional engineering. The list below is a planning checklist, not a claim
that these features already exist.

### Identity and access

- Add secure user authentication.
- Add organization or workspace membership.
- Define role-based permissions.
- Enforce authorization on every page and API operation.
- Protect ticket IDs from cross-tenant access.
- Support account recovery and session revocation.
- Add administrative access controls.
- Log security-relevant identity events.

### Tenant isolation

- Define a tenant identity and lifecycle.
- Scope every record by tenant.
- Scope database queries by authenticated tenant context.
- Isolate model artifacts by tenant.
- Isolate uploaded datasets by tenant.
- Isolate exports and experiment results by tenant.
- Test that one tenant cannot access another tenant's records.
- Define tenant deletion and export behavior.

### Hosted data management

- Replace local CSV path configuration with managed dataset ingestion.
- Validate uploads before accepting them.
- Enforce file type and size limits at the edge and application layers.
- Store source files in controlled object storage.
- Encrypt sensitive data at rest and in transit.
- Define retention and deletion policies.
- Record provenance and dataset versions.
- Add background processing for expensive evaluations.
- Implement database migrations and backups.
- Test backup restoration.

### Runtime and reliability

- Use a production WSGI server.
- Terminate HTTPS at an appropriate gateway.
- Use managed secret storage.
- Configure health checks and readiness checks.
- Add structured logs with sensitive-value filtering.
- Add error reporting and service metrics.
- Define request timeouts and resource limits.
- Use a job queue for long-running experiments.
- Add idempotent job execution and cancellation.
- Define operational support and incident response.

### Product experience

- Add organization settings and member management.
- Add user-specific preferences.
- Add saved views and tenant-level filters.
- Add durable notifications for completed jobs.
- Add accessible loading, error, and empty states.
- Add documented data export and deletion workflows.
- Add localization and timezone behavior if required.
- Define supported integrations and their permission scope.
- Keep human approval visible for consequential actions.

### Security and compliance

- Perform threat modeling.
- Conduct dependency and application security reviews.
- Add rate limits and abuse monitoring.
- Validate and encode all user-controlled inputs.
- Review file uploads and generated exports.
- Establish data classification and access policies.
- Document subprocessors and external services if introduced.
- Define privacy notices and retention statements.
- Test restore, deletion, and access-control workflows.
- Obtain appropriate legal and compliance review before launch.

## Glossary

### Abstention

A model decision to avoid assigning a known label when configured evidence
rules indicate insufficient support.

### Accuracy

The number of correct predictions divided by the total number of evaluated
predictions.

### Active learning

A workflow that selects examples for human labeling to improve a model or
research process.

### AUPR

Area under the precision–recall curve for a specified positive class.

### AUROC

Area under the receiver operating characteristic curve.

### Calibration

Agreement between a model's reported confidence and empirical correctness.

### Class support

The number of evaluated examples associated with a class label.

### Confidence

A score emitted by a model. In this application, it may be uncalibrated.

### Counterfactual wording

An alternate text scenario used to observe model output sensitivity; not
necessarily a causal intervention.

### Dataset fingerprint

A digest and related metadata used to identify a specific loaded dataset
state.

### Exact-text group

Records whose normalized text is identical under the grouping function.

### F1 score

The harmonic mean of precision and recall.

### Holdout

A partition reserved for evaluation and not used to fit the corresponding
model.

### Idempotency key

A request identifier used to recognize retries of a state-changing operation.

### Label overlay

A local human-approved ticket label that does not alter the source CSV.

### Majority baseline

A classifier that predicts the most frequent label observed in its training
partition.

### Near-duplicate

Text that is similar under a configured similarity measure but not identical
after exact normalization.

### Open-set recognition

The task of identifying inputs that do not belong to the classes used to fit
the classifier.

### Provenance

Documentation describing a dataset's origin, license, retrieval, and
verification state.

### Risk–coverage curve

A plot of selective prediction error versus the fraction of cases accepted.

### SaaS

Software as a Service: software operated as an online service for users,
typically with managed identity, tenancy, hosting, and operations.

### Selective risk

Error rate among the subset accepted by a selective prediction policy.

### TF-IDF

Term Frequency–Inverse Document Frequency, a statistical representation of
text used by the classifiers in this project.

### Tenant

An organization or customer boundary used to isolate hosted SaaS data and
permissions. Opslyra does not currently implement tenant isolation.

## FAQ

### Is Opslyra already a hosted SaaS product?

No. It is a locally run, single-workspace application with a SaaS-style
interface. It does not provide hosted user accounts, tenant isolation, or a
production deployment stack.

### Does it connect to a real ticket system?

No. It reads a configured CSV. It does not synchronize with a live service
desk or update external tickets.

### Does autonomous mode reply to customers?

No. Customer replies and external ticket actions are not connected or
attempted.

### Does approving a recommendation edit the CSV?

No. Approval writes a local label overlay and an activity event.

### Are prediction scores calibrated probabilities?

Not automatically. Scores shown by the base classifiers are uncalibrated
unless a separate calibration method has been evaluated and selected.

### Do the benchmark results prove the dataset labels are random?

No. The text baseline failing to exceed a majority baseline is a signal
finding, not proof of how the labels were generated.

### Does duplicate grouping eliminate leakage?

It prevents normalized exact-text groups from crossing the relevant split.
Near-duplicate but nonidentical text can still appear in different partitions.

### Are the routing and SLA results operational measurements?

No. Those workflows use configured assumptions or proxies because verified
team, capacity, processing-time, and SLA ground truth is unavailable.

### Why can a temporal result be unavailable?

The data may not contain enough distinct, parseable, trusted chronological
dates to support a meaningful holdout.

### Is the bundled dataset licensed by the repository's MIT license?

No. The dataset and third-party material are excluded from the source-code
license. Verify the original dataset terms before reuse or redistribution.

### Where are experiment outputs stored?

By default, CLI experiment JSON files are written under
`instance/experiments/`.

### How do I clear local state?

Stop the server, back up anything important, and remove only the specific
database, artifact, or experiment files you intend to reset.

### Which Python versions are tested in CI?

Python 3.10, 3.11, and 3.13 are in the current workflow matrix.

### How should I cite an annotation agreement result?

Use two independently produced annotation files and follow
`docs/ANNOTATION_GUIDE.md`. Agreement computed from non-independent labels is
not independent validation.

## License

Opslyra source code and project documentation are licensed under the MIT
License; see [LICENSE](LICENSE). The dataset and third-party materials are
excluded from that grant and remain subject to their own terms.

---

**Opslyra — From Every Ticket to Intelligent Action.**
