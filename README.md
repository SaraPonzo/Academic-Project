This repository collects all my academic projects, both individual and te am-based, completed during my MSc in *Financial Risk and Data Analysis* at Sapienza University of Rome.  


- **[GARCH vs Stochastic Volatility on ENI (2010–2024)](./garch-vs-sv-eni/)**: the project compares **GARCH-family** models (sGARCH, EGARCH, GJR) with **Stochastic Volatility (SV)** specifications that allow leverage and heavy-tailed errors. Using daily log returns, a one-step-ahead forecast was produced and the models evaluated via **AIC**, **RMSE** against a realized-volatility proxy, and **risk backtesting** (VaR/ES). Volatility regimes were with **PELT** changepoints and characterize tail behavior using **Extreme Value Theory (EVT)** by fitting **GEV** distributions to monthly block maxima/minima. 
  - Optional: direct HTML report: `./garch-vs-sv-eni/report/Caso_Ponzo_GARCHvsSV.html`

