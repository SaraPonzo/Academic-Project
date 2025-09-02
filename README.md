Welcome! This repository collects **all academic projects**—both individual and **team-based**—completed during my MSc in *Financial Risk and Data Analysis* at Sapienza University of Rome.  
Each folder contains code, reports, and notes for reproducibility. Co-authors are credited in the project READMEs and commit history.

## Included projects

- **GARCH vs Stochastic Volatility on ENI (2010–2024)** — [open project](./garch-vs-sv-eni/)  
  *Abstract.* This project compares **GARCH-family** models (sGARCH, EGARCH, GJR) with **Stochastic Volatility (SV)** specifications that allow leverage and heavy-tailed errors. Using daily log returns, we produce one-step-ahead forecasts and evaluate models via **AIC**, **RMSE** against a realized-volatility proxy, and **risk backtesting** (VaR/ES). We detect volatility regimes with **PELT** changepoints and characterize tail behavior using **Extreme Value Theory (EVT)** by fitting **GEV** distributions to monthly block maxima/minima. The pipeline is fully reproducible and links figures/tables to the report.
  - Optional: direct HTML report: `./garch-vs-sv-eni/report/Caso_Ponzo_GARCHvsSV.html`
