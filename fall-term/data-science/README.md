# Data Science — MMF1922H

**Master of Mathematical Finance — University of Toronto**  
**Fall 2026**  
**Instructor:** Charles Tsang

## Course Overview

This course introduces data science and machine learning methods applied to finance, investment, and risk management, combining statistical foundations with practical modelling and data-processing techniques.

The course covers the machine learning lifecycle, from problem formulation and data preparation to model evaluation, interpretability, and production considerations.

Topics include:

- Probability and statistical foundations
- Exploratory Data Analysis (EDA) and feature engineering
- Supervised learning: regression, classification, and ensemble methods
- Unsupervised learning: clustering and dimensionality reduction
- Financial data challenges: non-stationarity, class imbalance, and data leakage
- Big Data processing with PySpark
- MLOps, model monitoring, and concept/data drift
- Model evaluation, interpretability, and communication of results

## Final Project — Explainable Credit Risk

**Type:** Team Project (4 members)  
**Focus:** Explainable AI, credit risk modelling, model interpretability

[**View Project Repository — Explainable Credit Risk**](https://github.com/jaimecrz3/explainable-credit-risk)

The final project investigates how different machine learning models can be interpreted in the context of credit default prediction, with particular emphasis on the reliability and limitations of Explainable AI (XAI) techniques.

Using the UCI Default of Credit Card Clients dataset, we examine three classification models:

- **Logistic Regression:** Direct interpretation through model coefficients.
- **XGBoost:** Global feature importance and differences between importance measures.
- **Neural Network:** Local explanations using SHAP, LIME, and DiCE counterfactuals.

### Key Findings

The analysis highlights that model explanations are not necessarily consistent or actionable:

- Feature-importance rankings can change depending on the measure used.
- SHAP and LIME may provide different explanations for the same prediction.
- LIME explanations can vary across repeated runs.
- Counterfactual explanations may suggest changes that are mathematically valid but financially unrealistic.

**Main takeaway:** No single explanation method provides a complete understanding of a model. In financial applications, explanation reliability, stability, and practical feasibility matter alongside predictive performance.

The [project repository](https://github.com/jaimecrz3/explainable-credit-risk) contains the model implementations, explainability experiments, results, and final team presentation.