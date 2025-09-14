1) Project Overview

This repository contains the work for my master’s thesis, which focused on predicting fall risk in patients with Multiple Sclerosis (MS) using clinical data and machine learning models. The aim was not only to build predictive models, but also to understand which clinical factors most strongly contribute to fall risk and how these findings could support real-world clinical practice.

2)Data Source

The dataset was obtained from the MSOAC (Multiple Sclerosis Outcome Assessments Consortium) Placebo Database, which is publicly available through the Critical Path Institute: https://c-path.org/multiple-sclerosis-outcome-assessments-consortium-msoac-placebo-database-faq/

The database includes anonymized clinical data such as:

EDSS (Expanded Disability Status Scale)

KFSS (Kurtzke Functional Systems Scores) – including cerebellar, sensory, pyramidal, and bowel/bladder subscores

Demographic variables (age, sex, race)

Comorbidities (e.g., hypertension, arthritis, diabetes)

3) Methodology & Workflow

The project was developed step-by-step:

Data Preprocessing

Cleaned and structured raw clinical data.

Created derived features such as EDSS bins and KFSS subscore groupings.

Stored and managed data efficiently in MongoDB for scalable handling.


4)Exploratory Analysis

-Visualized fall rates across different disability levels (e.g., EDSS bins, KFSS domains).

- Identified patterns, such as higher fall rates linked to cerebellar and pyramidal dysfunction.

5) Modeling

- Two models were selected deliberately for complementary reasons:

Logistic Regression (LR): A simple, interpretable baseline that shows how each variable contributes to fall risk.

Histogram Gradient Boosting (HGB): A more advanced model capable of capturing non-linear interactions and handling imbalanced data more effectively.


6)Evaluation

-Metrics: AUC, specificity, sensitivity, accuracy, F1-score.

-Special focus on specificity (90% target) to reduce false alarms in clinical use.

- Analysis included ROC curves, confusion matrices, and permutation importance.

-Key Results

Logistic Regression:

Provided transparency into key features.

Top predictors: cerebellar dysfunction, bowel/bladder dysfunction, age, sex.

Moderate discriminative power (AUC ≈ 0.65).

Histogram Gradient Boosting:

Improved predictive performance (AUC ≈ 0.75).

Achieved higher specificity (~0.97), but sensitivity remained limited (~0.43).

Better at distinguishing non-fallers, but still struggled to capture the small group of actual fallers.

Threshold Analysis:

Setting a threshold at ~0.71 balanced specificity and sensitivity.

Prioritizing 90% specificity minimized false positives but missed some true fallers — reflecting the real-world trade-off clinicians face.

7) Why These Models

Logistic Regression: Chosen for interpretability and clinical trust. Doctors can see exactly which factors contribute to risk.

Histogram Gradient Boosting: Chosen for its ability to handle imbalanced data and capture complex patterns missed by simpler models.

The combination highlighted a central theme: interpretability vs predictive power.

8) Examples of Outputs

ROC Curve (Logistic Regression) showed moderate separation between fallers and non-fallers.

Confusion Matrices demonstrated that both models often missed fallers (false negatives), but HGB significantly reduced false positives.

Feature Importance Plots validated that dysfunction in cerebellar and bowel/bladder systems were among the strongest clinical predictors.

9) Tools & Libraries

Python (scikit-learn, matplotlib, pandas, numpy)

MongoDB (for data management)

LaTeX (for thesis preparation and structured documentation)

10) Conclusion

This project demonstrates both the potential and limitations of using clinical data for fall-risk prediction in MS. While models like Histogram Gradient Boosting improved accuracy, the challenge of detecting true fallers remains significant. Logistic Regression, despite weaker performance, provided valuable interpretability and clinical insights.

The findings emphasize the importance of balancing predictive accuracy with clinical usability when designing AI tools for healthcare.
