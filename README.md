# 🎓 MSc Academic Projects — Financial Risk & Data Analysis (Sapienza)

Welcome! This repository collects **my academic projects**, both individual and **team based**,  completed during my MSc in *Financial Risk and Data Analysis* at Sapienza University of Rome.  


## 📁 Projects

- **[GARCH vs Stochastic Volatility on ENI (2010–2024)](./garch-vs-sv-eni/)**  
  *Abstract.* This project compares **GARCH-family** models (sGARCH, EGARCH, GJR) with **Stochastic Volatility (SV)** specifications (leverage + heavy-tailed errors). Using daily log returns, we produce one-step-ahead forecasts and evaluate models via **AIC**, **RMSE** against a realized-volatility proxy, and **risk backtesting** (VaR/ES). We detect volatility regimes with **PELT** changepoints and characterize tail behavior using **Extreme Value Theory (EVT)** by fitting **GEV** distributions to monthly block maxima/minima.  
  *Report:* `./garch-vs-sv-eni/report/Caso_Ponzo_GARCHvsSV.html`

- **[Conformal Inference — Distribution-Free Uncertainty Quantification](./conformal-inference/)**  
  *Abstract.* This project implements **split/inductive conformal prediction** to obtain distribution-free **(1 − α)** coverage under exchangeability. For regression we use **Conformalized Quantile Regression (CQR)** to handle heteroscedastic noise; for classification we build **set-valued (top-k)** predictions from softmax scores. We include **Mondrian (class-conditional)** and **normalized** nonconformity scores to stabilize interval widths across strata, plus rolling/blocked variants for **time-series**. Evaluation covers **empirical coverage**, **interval width / set size**, and the **efficiency–coverage trade-off**, with stress tests under covariate shift.
  Report: `./conformal-inference/report/Conformal_Inference.html`
