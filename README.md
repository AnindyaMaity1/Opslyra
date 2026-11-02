# Opslyra

This project is built under the guidance of **Prof. Dr. Govind Pandey.** 

LinkedIn Profile -  [Link](https://www.linkedin.com/in/dr-govind-pandey-145a43131/). 


Department of CSE (AI & ML), Dayananda Sagar University, Bengaluru, India.



**From Every Ticket to Intelligent Action.**


Opslyra is a local, research-oriented application designed to help explore customer-support ticket data, identify patterns, support ticket prioritization, and investigate AI-assisted workflows. It combines classical machine-learning models, analytics, and human-in-the-loop workflow simulations in an interactive dashboard.

## 1. Project Overview

Customer-support teams handle large volumes of tickets with varying urgency, categories, and resolution requirements. Opslyra provides a
prototype environment for examining how ticket analytics and machine learning can help organize this information and support human
decision-making.

The application loads ticket records from a configurable CSV dataset and
provides:

-   Dashboard summaries and ticket exploration.
-   Data-quality checks and descriptive analytics.
-   NLP-based ticket classification using classical machine-learning methods.
-   AI-assisted workflow modes with human oversight.
-   Research-oriented experiments for novelty, similarity, routing, and model diagnostics.
-   Local persistence for selected application records and experiments.

Opslyra is intended for local research, demonstration, and
experimentation. It is not a production customer-support platform.

## 2. Research Motivation

The project explores how AI-assisted ticket workflows can remain useful while keeping decisions transparent and reviewable. Its main themes include:

-   Converting unstructured ticket text into useful analytical signals.
-   Supporting ticket categorization and prioritization.
-   Examining when a model's predictions may be uncertain or unfamiliar.
-   Simulating workflow recommendations while preserving human control.
-   Investigating ticket similarity, routing constraints, and operational patterns.
-   Communicating model behavior through dashboards and diagnostics.

The prototype emphasizes decision support rather than fully automated customer-facing actions.

## 3. Core Features

### 3.1 Ticket analytics and dashboard

The dashboard supports ticket-level exploration and operational summaries, including ticket distributions, status and category views, and dataset-quality indicators. Runtime statistics are derived from the dataset loaded by the application.

### 3.2 NLP-based machine-learning classification

Opslyra uses a TF-IDF text representation with classical machine-learning classifiers, including Logistic Regression, to explore ticket-label prediction.

The model uses redacted subject and description text as its primary lexical inputs. Classification outputs are intended as exploratory predictions and should be reviewed by a human.

### 3.3 Adaptive AI workflow modes

The application demonstrates several workflow modes:

-   **Fully Manual** --- human users make workflow decisions.
-   **Copilot** --- AI provides recommendations for human review.
-   **Selective Automation** --- selected workflow steps are simulated under configured conditions.
-   **Autonomous Workflow** --- demonstrates a more automated decision path within the local prototype.
-   **Human Takeover** --- prioritizes human control and blocks new AI workflow activity where implemented.

These modes are local workflow demonstrations. They do not independently contact customers, send messages, or execute external business actions.

### 3.4 Incident Lab and research workflows

Incident Lab provides a collection of research-oriented experiments, including:

-   Human label corrections, training, and model-version exploration.
-   Lexical novelty and out-of-distribution-style checks.
-   Counterfactual scenario exploration.
-   Ticket-similarity graph analysis.
-   SLA-risk sandboxing.
-   Simulated ticket routing.
-   Trend and cluster exploration.
-   Pattern-based text redaction.
-   Similar resolved-ticket retrieval.
-   Model diagnostics and evaluation views.

The experiments are intended to support investigation and demonstration.

### 3.5 Human oversight and decision support

Opslyra is designed around reviewable recommendations. Human corrections and takeover concepts help demonstrate how model outputs can be inspected or overridden rather than treated as authoritative decisions.

## 4. System Architecture

The project follows a modular local application architecture:

-   **Flask application** --- Web interface and application routes.
-   **Templates and static assets** --- Dashboard pages, ticket views, and frontend styling.
-   **Analytics and workflow logic** --- Ticket summaries, recommendations, and experiment orchestration.
-   **Modeling components** --- Text vectorization, classification, and diagnostic evaluation.
-   **Research modules** --- Novelty, similarity, counterfactual, routing, and simulation workflows.
-   **SQLite** --- Local persistence for supported application records and experiments.
-   **CSV dataset** --- Primary ticket-data input.

Typical application flow:

1.  Load the configured ticket dataset.
2.  Validate available fields and inspect data quality.
3.  Generate dashboard summaries and ticket views.
4.  Transform ticket text into TF-IDF features.
5.  Train or use supported classification models.
6.  Explore recommendations and research workflows.
7.  Review results through the local dashboard.


## 5. Data Model and Inputs

Opslyra expects a customer-support ticket dataset in CSV format.

The application is designed around fields commonly found in ticket datasets, such as:

-   Ticket identifier. 
-   Customer or requester information. 
-   Ticket subject and description. 
-   Ticket type or category. 
-   Priority. 
-   Status. 
-   Channel. 
-   Product or service. 
-   Resolution information. 
-   Creation and response timestamps. 
-   SLA-related fields. 

The exact fields available depend on the dataset. The application should be used with data that the user is authorized to process.

## 6. AI Workflow and Evaluation

The classification workflow uses TF-IDF features derived from ticket text and classical machine-learning estimators. Evaluation and
diagnostics are intended to help inspect model behavior during experimentation.


## 7. Scientific and Practical Limitations

Opslyra is a research prototype with the following limitations:

-   It is designed for local experimentation, not production deployment.
-   Model performance depends on the supplied dataset and its labels.
-   TF-IDF features capture lexical patterns and may miss context or meaning.
-   SLA-risk and routing experiments use simplified assumptions.
-   Counterfactual scenarios are exploratory and do not establish causality.


Before using the system with real operational data, review data governance, privacy requirements, security controls, model validation, and deployment architecture.

## 9. Example Use Cases

Opslyra can be used for:

-   Research prototypes in AI-assisted customer support.
-   Exploring ticket classification and operational analytics.
-   Demonstrating human-in-the-loop AI workflows.
-   Investigating ticket similarity and lexical novelty.
-   Comparing model behavior on a selected dataset.
-   Teaching applied machine learning and responsible AI concepts.
-   Exploring simulated routing and SLA-risk scenarios.


## 12. Summary

Opslyra demonstrates a local workflow for customer-support ticket intelligence by combining:

-   Ticket analytics and data-quality inspection.
-   TF-IDF-based machine-learning classification.
-   Adaptive AI workflow demonstrations.
-   Human oversight and takeover concepts.
-   Research experiments for novelty, similarity, routing, and counterfactual analysis.
-   Dashboard-based exploration and model diagnostics.

Opslyra is best understood as a transparent, research-oriented prototype for exploring AI-assisted ticket operations. Further work should focus on dataset validation, stronger privacy and security controls, rigorous
evaluation, and carefully governed integration with real support systems.