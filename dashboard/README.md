## Dashboard Features & Architecture

Interactive Spatial Risk Atlas: Built via Streamlit to bridge the machine learning pipeline and public health practice, integrating predictive models with county-level Social Determinants of Health data.

Dual-Audience Design: Delivers actionable insights for public health officials alongside deep statistical validation tools for technical auditors.

Geographic Exploration & Benchmarking: Features state-level drop-down menus, raw master dataset previews, and independent County Health Rankings (CHR&R/CDC) metrics included as benchmarks without causing model multicollinearity.

Auditor Verification Tabs: Provides dedicated views for exploratory data analysis (EDA), including distributions, correlation heatmaps, model performance metrics, and an interactive model explorer to live-train and verify alternative regression models against XGBoost.

SHAP-Driven Predictive Insights: Equips stakeholders with feature impact tools to inspect key drivers of hypertension prevalence by county.

Integrity & Fairness Framework:

    - Uses aggregate federal data requiring no de-identification.

    - Visually separates model forecasts from raw baseline measurements.

    - Explicitly frames structural metrics (such as food access) as systemic inequities rather than modifiable intervention targets, honoring that the analytics display association rather than causality.

🔗 Live App: [<ads599capstonehypertensionriskatlas-3tepewveftselpfgfsn9nc.streamlit.app>](https://ads599capstonehypertensionriskatlas-b6cz6ygt7lui5xt4s4awuk.streamlit.app/)