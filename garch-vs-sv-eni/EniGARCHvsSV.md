

##  Make a comparison between GARCH and stochastic volatility models.

To accomplish the task we have decided to analyse the behavior of ENI
over the 2010-2024 period by:

1.  Fitting 3 types of GARCH : standard, exponential and gjr.

2.  Fitting 2 types of stochastic volatility model: leverage and
    t-distribution.

3.  Verify the robustness of the model by analyzing the volatility
    clusters.

4.  Make the comparison between all the models through the AIC and the
    RMSE to verify which model better forcast the future volatility.

## Why did we choose these types of models?

We have chosen to analyse the behavior of ENI because of the main
patterns of the energy market, which is usually exposed to volatility
clusters due to geopolitical risks and oil prices. Additionally,
leverage effects and co-movements with commodities have a strong
influence over the energy price too, making the negative news have a
bigger influence on volatility.

Both garch and stochastic volatility models can be useful to recognize
these features and when using these models we have to take into account
both the strenghts and limitations they have.

We included the EGARCH and GJR-GARCH models because they allow us to
capture asymmetric volatility responses, which are crucial in the energy
market.

EGARCH models volatility in logarithmic form,removing the need to impose
positivity constraints and allowing shocks of different signs to have
different effects.

GJR-GARCH, on the other hand, introduces an indicator function that
amplifies the effect of negative shock, which are very common in the
energy market due to the leverage effect.

For the stochastic volatility models, the inclusion of a version with a
Student’s t-distribution is essential to account for extreme market
model more realistically, something the gaussian’s models are not able
to do.

**When to use GARCH models?**

- To forecast short-term volatility;

- When we have high-frequency financial returns, like daily/hourly
  equity or FX data and conditional heteroskedasticity is pronounced;

- To have quick estimation and interpretable estimation, often using
  maximum likelihood.

**Limitations**

- They assume volatility is a deterministic function of past data;

- They are not ideal for capturing long memory or latent volatility
  shocks;

- Poorer fit for realistic volatility persistence and non-linearities.

**When to use stochastic volatility models?**

- To capture latent and persistent volatility. In fact, SV models treat
  volatility as a hidden stochastic process, better reflecting reality
  compared to Garch models;

- To model financial assets with strong regime shifts;

- To handle heavy tails and jumps. SV models with t-distribution or jump
  components model fat tails and sudden shock better.

**Limitations**

- Harder to compute, sometimes requiring MCMC or particle filters;

- Harder to implement in large-scale applications or real-time trading;

- Less intuitive.

## 1. GARCH estimation for ENI.

We upload the needed packages:

``` r
library(quantmod)
```

    ## Warning: il pacchetto 'quantmod' è stato creato con R versione 4.4.3

    ## Caricamento del pacchetto richiesto: xts

    ## Caricamento del pacchetto richiesto: TTR

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
library(tidyverse)
```

    ## Warning: il pacchetto 'tidyverse' è stato creato con R versione 4.4.3

    ## Warning: il pacchetto 'readr' è stato creato con R versione 4.4.3

    ## Warning: il pacchetto 'forcats' è stato creato con R versione 4.4.3

    ## ── Attaching core tidyverse packages ──────────────────────────────────────────────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ## ✔ ggplot2   3.5.1     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.3     ✔ tidyr     1.3.1
    ## ✔ purrr     1.0.2

    ## ── Conflicts ────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::first()  masks xts::first()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ✖ dplyr::last()   masks xts::last()
    ## ✖ purrr::reduce() masks rugarch::reduce()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(zoo)
library(quantmod)
library(rugarch)
library(stochvolTMB)
```

    ## Warning: il pacchetto 'stochvolTMB' è stato creato con R versione 4.4.3

    ## 
    ## Caricamento pacchetto: 'stochvolTMB'
    ## 
    ## Il seguente oggetto è mascherato da 'package:rugarch':
    ## 
    ##     residuals
    ## 
    ## Il seguente oggetto è mascherato da 'package:changepoint':
    ## 
    ##     residuals
    ## 
    ## Il seguente oggetto è mascherato da 'package:stats':
    ## 
    ##     residuals
    ## 
    ## Il seguente oggetto è mascherato da 'package:utils':
    ## 
    ##     demo

``` r
library(data.table)
```

    ## Warning: il pacchetto 'data.table' è stato creato con R versione 4.4.3

    ## data.table 1.17.0 using 10 threads (see ?getDTthreads).  Latest news: r-datatable.com
    ## 
    ## Caricamento pacchetto: 'data.table'
    ## 
    ## I seguenti oggetti sono mascherati da 'package:lubridate':
    ## 
    ##     hour, isoweek, mday, minute, month, quarter, second, wday, week, yday, year
    ## 
    ## I seguenti oggetti sono mascherati da 'package:dplyr':
    ## 
    ##     between, first, last
    ## 
    ## Il seguente oggetto è mascherato da 'package:purrr':
    ## 
    ##     transpose
    ## 
    ## I seguenti oggetti sono mascherati da 'package:xts':
    ## 
    ##     first, last
    ## 
    ## I seguenti oggetti sono mascherati da 'package:zoo':
    ## 
    ##     yearmon, yearqtr

``` r
library(ggplot2)
library(reshape2)
```

    ## 
    ## Caricamento pacchetto: 'reshape2'
    ## 
    ## I seguenti oggetti sono mascherati da 'package:data.table':
    ## 
    ##     dcast, melt
    ## 
    ## Il seguente oggetto è mascherato da 'package:tidyr':
    ## 
    ##     smiths

``` r
library(changepoint)
library(patchwork)
```

    ## Warning: il pacchetto 'patchwork' è stato creato con R versione 4.4.3

``` r
library(ismev)
```

    ## Warning: il pacchetto 'ismev' è stato creato con R versione 4.4.3

    ## Caricamento del pacchetto richiesto: mgcv
    ## Caricamento del pacchetto richiesto: nlme
    ## 
    ## Caricamento pacchetto: 'nlme'
    ## 
    ## Il seguente oggetto è mascherato da 'package:dplyr':
    ## 
    ##     collapse
    ## 
    ## This is mgcv 1.9-1. For overview type 'help("mgcv-package")'.

We upload the time series of ENI from 2010 to 2024 using the “quantmod”
package. Specifically we use the getSymbols() function. We computed the
daily log returns based on the closing prices of ENI, multiplied by 100
to express them in percentage terms.

``` r
getSymbols("ENI.MI", from = "2010-01-01", to = "2024-12-31")
```

    ## [1] "ENI.MI"

``` r
eni = `ENI.MI`
returns = na.omit(100 * diff(log(Cl(eni))))
```

In this part we use a 21-day rolling window (approximately one trading
month) to compute the realized volatility. The object ‘align = “right”’
ensures that each value reflects the past 21 days ending at that point,
which is ideal for forecasting and visualization. Additionally, we use
the “fortify.zoo()” function to convert the zoo object into a data frame
with a date column for plotting.

``` r
realized_vol <- rollapply(returns, width = 21, FUN = sd, fill = NA, align = "right")
colnames(realized_vol) <- "RealizedVol"  
vol_df <- fortify.zoo(realized_vol)
colnames(vol_df) <- c("Date", "RealizedVol")
```

Then, we ploy the realized rolling volatility to make a first visual
check.

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/pressure-1.png)<!-- -->

From the plot we can see that volatility is not constant, in fact it
clusters in time which is ideal for using these types of models.
Multiple clusters can be observed throught time:a high spike around
march-april 2020 due to the Covid-19 pandemic, some other local smaller
spikes around 2011-2012 due to the european sovereign debt crisis and
around 2022 due to the russian invasion of Ukraine.

Using the data from 2010 to 2024, we will forecast the volatility for
the 60 days ahead. After that, we will use the realized volatility for
the first two months of the 2025 and verify the ability of the four
model to forecast future volatility.

Volatility in GARCH models is **deterministically driven** by past
squared returns and past volatility. In its standard form, the
**GARCH(1,1)** model is written as:

$$
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
$$

Where:

- $\sigma_t^2$ is the conditional variance at time $t$.

- $\epsilon_{t-1}$ is the return shock (innovation) at time $t-1$.

- $\alpha_0 > 0$, $\alpha_1 \geq 0$, and $\beta_1 \geq 0$ are
  parameters.

The model implies volatility depends on **past shocks and past
volatility**.

### EGARCH and GJR-GARCH Extensions

Extensions like **EGARCH** (Exponential GARCH) and **GJR-GARCH** are
designed to capture **asymmetry or leverage effects**.

#### GJR-GARCH(1,1) Specification:

$$
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \gamma_1 \epsilon_{t-1}^2 \cdot I_{\{\epsilon_{t-1} < 0\}} + \beta_1 \sigma_{t-1}^2
$$

- $I_{\{\epsilon_{t-1} < 0\}}$ is an indicator function equal to 1 when
  the shock is negative.

#### EGARCH(1,1) Specification:

$$
\log(\sigma_t^2) = \omega + \beta \log(\sigma_{t-1}^2) + \alpha \frac{\epsilon_{t-1}}{\sigma_{t-1}} + \gamma \left( \left| \frac{\epsilon_{t-1}}{\sigma_{t-1}} \right| - \mathbb{E}\left| \frac{\epsilon_{t-1}}{\sigma_{t-1}} \right| \right)
$$

- This formulation allows for **asymmetric effects** and guarantees
  non-negative variance **without needing parameter restrictions**.

### Distributional Assumptions

GARCH models typically assume that standardized residuals follow:

- A **normal distribution**,

or

- A **t-distribution** to better capture **fat tails** in financial
  returns.

Using the **t-distribution** we can model heavy tails than simply
adjusting residuals in a normal GARCH.

------------------------------------------------------------------------

**a. GARCH (1,1)**

We start from the ‘ugarchspec’ function included in the ‘rugarch’
package in which we specify the type of GARCH we want to compute, the
standard GARCH.

Then we fit the model to the log returns and forecast 60 steps ahead and
extract the forecasted sigma path.

``` r
garch_spec = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0), include.mean = FALSE)
)
garch_fit = ugarchfit(spec = garch_spec, data = returns)
garch_forecast = ugarchforecast(garch_fit, n.ahead = 60)
garch_vol = sigma(garch_forecast)
```

**b. e-GARCH**

Here we specify that we want to compute an eGARCH(1,1) which models the
log of variance, ensuring positivity without parameter constraints and
captures asymmetric effects.

``` r
egarch_spec = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0), include.mean = FALSE)
)
egarch_fit = ugarchfit(spec = egarch_spec, data = returns)
egarch_forecast = ugarchforecast(egarch_fit, n.ahead = 60)
egarch_vol = sigma(egarch_forecast)
```

**c. gjr -GARCH**

Finally, we compute the gjr-GARCH that accounts better for negative
shocks.

``` r
gjr_spec = ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0), include.mean = FALSE)
)
gjr_fit = ugarchfit(spec = gjr_spec, data = returns)
gjr_forecast = ugarchforecast(gjr_fit, n.ahead = 60)
gjr_vol = sigma(gjr_forecast)
```

## 2. Stochastic volatility Models estimation for ENI.

Differently from GARCH, stochastic volatility (SV) models treat
volatility as an unobserved (latent) stochastic process that evolves
over time:

$$
\log(\sigma_t^2) = \mu + \phi \log(\sigma_{t-1}^2) + \eta_t
$$

Volatility follows a process with random innovations terms $\eta_t$,
making it more flexible and realistic in capturing features like long
memory, sudden shifts, and persistent clustering.

This structural difference explains why SV models often outperform GARCH
models in modeling real market volatility, especially over longer
horizons or during unstable economic cycles.

We will estimate and forecast volatility using two SV models: one with
leverage effects and one with Student t distribution.

**a. Leverage**

We use the estimate_parameters() function specifyng the type of SV model
we want to compute and then predict() to forecast the volatility.

``` r
sv_lev = estimate_parameters(as.numeric(returns), model = "leverage", silent = TRUE)
sv_lev_pred = predict(sv_lev, steps = 60, include_parameters = TRUE)
sv_lev_vol = summary(sv_lev_pred)$h_exp$mean
```

**b. t-distribution**

We do the same for the t-distribution SV model.

``` r
sv_t = estimate_parameters(as.numeric(returns), model = "t", silent = TRUE)
sv_t_pred = predict(sv_t, steps = 60, include_parameters = TRUE)
sv_t_vol = summary(sv_t_pred)$h_exp$mean
```

## 3. Verify the robustness of the model by analyzing the volatility clusters.

**Predictions**

We start by analysing the predictions of the different models.

``` r
result_df = data.frame(
  Day = 1:60,
  GARCH = as.numeric(garch_vol),
  EGARCH = as.numeric(egarch_vol),
  GJR = as.numeric(gjr_vol),
  SV_Leverage = sv_lev_vol,
  SV_t = sv_t_vol
)

melted = melt(result_df, id.vars = "Day")

ggplot(melted, aes(x = Day, y = value, color = variable)) +
  geom_line(size = 1.2) +
  labs(title = "Volatility forecast (60 days)",
       x = "Day", y = "Forecasted volatility",
       color = "Models") +
  theme_minimal()
```

    ## Warning: Using `size` aesthetic for lines was deprecated in ggplot2 3.4.0.
    ## ℹ Please use `linewidth` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was generated.

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The predictions highlight an increase in volatility over the next 60
days for all the considered models.

However, some significant differences can be observed in the
trajectories:

- the standard GARCH model shows a moderate and constant increase in
  volatility over time.

- the EGARCH model suggests a more rapid initial increase in volatility
  compared to the standard GARCH, indicating a potentially greater
  sensitivity to shocks.

- the GJR GARCH model exhibits a similar trend to the EGARCH in the
  initial phase but shows a tendency to stabilize slightly towards the
  end of the forecast horizon. This could reflect an asymmetric response
  to negative shocks compared to positive ones.

- the Stochastic Volatility with Leverage model (SV_Leverage) predicts a
  more pronounced increase in volatility compared to the GARCH models,
  suggesting that the leverage effect plays a significant role in the
  predicted dynamics.

- the Stochastic Volatility with t-distribution model (SV_t) shows a
  similar trend to the SV with leverage, but with generally higher
  volatility levels. The use of a t-distribution might better capture
  the presence of fat tails in the returns, leading to higher volatility
  forecasts.

In summary, while all models seems to predict an increase in volatility,
the stochastic volatility models tend to predict higher volatility
levels and potentially faster growth compared to the traditional GARCH
models.

However, to avoids interpreting volatility forecasts without a
context,it is important to identify the eventualvolatility clusters,
which provide a meaningful temporal framework for evaluating the models
and their ability to forecasts alongside RMSE and AIC.

We can perform a ‘changepoint’ analysis using the PELT method which
identifies several statistically significant change points in the
variance of the returns in the data. These change points mark the
transitions between periods of different volatility levels.

The segments extracted from the changepoint analysis can be used to
validate model predictions: if a model predicts elevated volatility
during a cluster labeled as “high volatility” (or viceversa) its output
aligns with the empirical classification and highlight a robust model.
This qualitative consistency reinforces the quantitative metrics like
AIC and RMSE.

``` r
y <- as.numeric(returns)
cpt <- cpt.var(y, method = "PELT", penalty = "MBIC")
summary(cpt)
```

    ## Created Using changepoint version 2.3 
    ## Changepoint type      : Change in variance 
    ## Method of analysis    : PELT 
    ## Test Statistic  : Normal 
    ## Type of penalty       : MBIC with value, 24.73694 
    ## Minimum Segment Length : 2 
    ## Maximum no. of cpts   : Inf 
    ## Changepoint Locations : 403 469 955 1199 1766 2574 2604 2795 3024 3411

``` r
plot(cpt, main = "Change points of the variance of returns")
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
changepoints <- cpts(cpt)
print(changepoints)
```

    ##  [1]  403  469  955 1199 1766 2574 2604 2795 3024 3411

The fact that we find multiple statistically significant change points
within the observed period further supports the presence of volatility
clustering.

The ability of a model to adapt to such shifts is crucial for
forecasting. Models like GJR-GARCH and EGARCH, which explicitly capture
asymmetric responses to shocks, are expected to better reflect such
transitions.

However, SV models are able to have even better results in this
scenario, especially the one with a Student t distribution errors, which
not only allow for regime-dependent behavior but also model the
uncertainty about volatility itself.

In the following part we will try to extract the standard deviation for
different ranges of periods and the corresponding type of volatility
(high or low).

We start by defining a vector called ‘breaks’ to encapsulate all the
different change point. This is important to divide the time series in
segments based on the change points. We created a for loop to iterate
through the breaks vector to extract the subset of the returns
corresponding to each segment.

If the extracted standard deviation is bigger than the standard
deviation of the returns the assigned status is “high volatility”,
viceversa for standard deviations smaller than the general one.

``` r
cp_index <- cpts(cpt)
breaks <- c(1, cp_index, length(y))  
global_sd <- sd(y)
cluster_list <- list()

for (i in 1:(length(breaks) - 1)) {
  start <- breaks[i]
  end <- breaks[i + 1]
  segment <- y[start:end]
  segment_sd <- sd(segment)
  status <- ifelse(segment_sd > global_sd, "High volatility", "Low volatility")
  cluster_list[[i]] <- data.frame(
    Start = index(returns)[start],
    End = index(returns)[end],
    SD = round(segment_sd, 4),
    Cluster = status
  )
}

cluster_df <- do.call(rbind, cluster_list)
print(cluster_df)
```

    ##         Start        End     SD         Cluster
    ## 1  2010-01-05 2011-07-29 1.3685  Low volatility
    ## 2  2011-07-29 2011-11-01 2.8610 High volatility
    ## 3  2011-11-01 2013-10-01 1.5055  Low volatility
    ## 4  2013-10-01 2014-09-19 1.0871  Low volatility
    ## 5  2014-09-19 2016-12-12 1.9761 High volatility
    ## 6  2016-12-12 2020-02-21 1.0737  Low volatility
    ## 7  2020-02-21 2020-04-03 7.1895 High volatility
    ## 8  2020-04-03 2021-01-06 2.5571 High volatility
    ## 9  2021-01-06 2021-11-25 1.2836  Low volatility
    ## 10 2021-11-25 2023-06-02 1.8788 High volatility
    ## 11 2023-06-02 2024-12-30 1.1167  Low volatility

Comparing the average predicted volatility of each model to the standard
deviation threshold used in the changepoint clustering, we find that all
forecasts remain within the range of low-volatility cluster.

``` r
forecast_means <- colMeans(result_df[, -1]) 
forecast_classification <- ifelse(forecast_means > global_sd, 
                                  "High volatility forcasted", 
                                  "Low volatility forcasted")

forecast_df <- data.frame(
  Models = names(forecast_means),
  Forcasted_mean = round(forecast_means, 4),
  Cluster = forecast_classification
)

print(forecast_df)
```

    ##                  Models Forcasted_mean                  Cluster
    ## GARCH             GARCH         1.4016 Low volatility forcasted
    ## EGARCH           EGARCH         1.4311 Low volatility forcasted
    ## GJR                 GJR         1.3749 Low volatility forcasted
    ## SV_Leverage SV_Leverage         1.4056 Low volatility forcasted
    ## SV_t               SV_t         1.2852 Low volatility forcasted

This outcome aligns with the final cluster identified by our analysis
It’s possible to notice how all the four models, despite their differing
internal mechanisms (deterministic vs latent), anticipate a continuation
of the low regime.

The agreement between historical clustering and forecasted volatility
levels supports the robustness of the modeling framework.

## 4. Comparison between the models through the AIC and the RMSE.

**AIC: akaike information criterion**

The AIC is used to compare different statistical models and assess their
goodness of fit while penalizing model complexity (number of
parameters). A lower AIC generally indicates a better model in terms of
balancing fit and parsimony.

To make a consistent comparison we have decided to calculate the AIC of
the GARCH models using the formula and not the direct function AIC(). In
this way we have an explicit control over the number of parameters to
use.

``` r
k_garch <- 3  
k_egarch <- 3  
k_gjr <- 3  
ll_garch  <- garch_fit@fit$LLH
ll_egarch <- egarch_fit@fit$LLH
ll_gjr    <- gjr_fit@fit$LLH

aic_garch  <- -2 * ll_garch + 2 * k_garch
aic_egarch <- -2 * ll_egarch + 2 * k_egarch
aic_gjr    <- -2 * ll_gjr + 2 * k_gjr

aic_lev<-AIC(sv_lev)
aic_t<-AIC(sv_t)

aic_df <- data.frame(
  Models = c("GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)", "SV Leverage", "SV t-distribuzione"),
  AIC = c(aic_garch, aic_egarch, aic_gjr, aic_lev, aic_t)
)

print(aic_df[order(aic_df$AIC), ]) 
```

    ##               Models      AIC
    ## 4        SV Leverage 13609.32
    ## 5 SV t-distribuzione 13626.71
    ## 3     GJR-GARCH(1,1) 13769.42
    ## 2        EGARCH(1,1) 13776.54
    ## 1         GARCH(1,1) 13841.29

- The SV Leverage model has the lowest AIC, suggesting it provides the
  best fit according to this criterion.

- The SV t-distribution model has the second-lowest AIC.

- The GARCH family models have higher AIC values, indicating a
  relatively poorer fit compared to the stochastic volatility models
  based on AIC.

**RMSE: root mean squared error**

The RMSE measures the average magnitude of the errors between 1 the
predicted values and the actual (realized) values. A lower RMSE
indicates that the predictions are closer to the actual values.

We use the rmse() function to compute it.

``` r
getSymbols("ENI.MI", from = "2025-01-01", to = "2025-03-31")
```

    ## [1] "ENI.MI"

``` r
eni_2025 <- `ENI.MI`
returns_2025 <- na.omit(100 * diff(log(Cl(eni_2025))))

returns_test <- as.numeric(returns_2025[1:60])

realized_vol_2025 <- sqrt(rollmean(returns_test^2, k = 2, align = "right", na.pad = FALSE))
realized_vol_2025 <- realized_vol_2025[1:59]

pred_garch <- result_df$GARCH[2:60]
pred_egarch <- result_df$EGARCH[2:60]
pred_gjr <- result_df$GJR[2:60]
pred_sv_lev <- result_df$SV_Leverage[2:60]
pred_sv_t <- result_df$SV_t[2:60]

rmse <- function(true, predicted) {
  sqrt(mean((true - predicted)^2, na.rm = TRUE))
}

rmse_values <- data.frame(
  Models = c("GARCH", "EGARCH", "GJR-GARCH", "SV Leverage", "SV t-dist"),
  RMSE = c(
    rmse(realized_vol_2025, pred_garch),
    rmse(realized_vol_2025, pred_egarch),
    rmse(realized_vol_2025, pred_gjr),
    rmse(realized_vol_2025, pred_sv_lev),
    rmse(realized_vol_2025, pred_sv_t)
  )
)

print(rmse_values[order(rmse_values$RMSE), ])
```

    ##        Models      RMSE
    ## 5   SV t-dist 0.7102274
    ## 3   GJR-GARCH 0.7640247
    ## 1       GARCH 0.7769960
    ## 4 SV Leverage 0.7861836
    ## 2      EGARCH 0.7990067

- The SV t-dist model has the lowest RMSE as it is expected, suggesting
  it provided then most accurate volatility forecasts compared to the
  realized volatility (calculated using a 2-day rolling window).

- The GJR-GARCH and GARCH models also show relatively low RMSE values.

- The SV Leverage and EGARCH models have higher RMSE values,indicating
  relatively less accurate predictions compared to the other models
  based on this metric.

# A visual comparison of the models results and the realized volatility

We analyze the latent volatility dynamics obtained from the stochastic
volatility models, comparing it with the rolling realized volatility for
the 21-day interval we previously estimated.

In fact, after estimating the models is it possible to calculate,
through the parameters produced by the summary, the conditional standard
deviation of returns, which permit us to make a direct comparison with
the Garchs results.

1.  **SV leverage model**

``` r
h_lev=summary(sv_lev)$estimate[-c(1:6)]
h_lev<-as.vector(h_lev)
h_lev<-h_lev[-c(1:2)]
sy_lev=summary(sv_lev)$estimate[1]
plot(sy_lev*exp(h_lev/2), type="l")
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

2.  **SV t-distribution model**

``` r
h_t=summary(sv_t)$estimate[-c(1:6)]
h_t<-as.vector(h_t)
h_t<-h_t[-c(1:2)]
sy_t=summary(sv_t)$estimate[1]
plot(sy_t*exp(h_t/2), type="l")
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
min_len_lev <- min(length(as.numeric(sy_lev*exp(h_lev/2))), length(realized_vol), length(returns))
df <- data.frame(
  Time = 1:min_len_lev,
  PredictedVol = (sy_lev*exp(h_lev/2))[1:min_len_lev],
  RealizedVol = realized_vol[1:min_len_lev]
)
p1 <- ggplot(df, aes(x = Time)) +
  geom_line(aes(y = PredictedVol, color = "Predicted Volatility (SV Model- Leverage)"), size = 1) +
  geom_line(aes(y = RealizedVol, color = "Realized Volatility (21-day Rolling SD)"), size = 0.5) +
  labs(
    title = "Predicted vs Realized Volatility",
    x = "Time",
    y = "Volatility",
    color = "Legend"
  ) +
  theme_minimal()

min_len_t <- min(length(as.numeric(sy_t*exp(h_t/2))), length(realized_vol), length(returns))
df <- data.frame(
  Time = 1:min_len_t,
  PredictedVol = (sy_t*exp(h_t/2))[1:min_len_t],
  RealizedVol = realized_vol[1:min_len_t]
)

p2 <- ggplot(df, aes(x = Time)) +
  geom_line(aes(y = PredictedVol, color = "Predicted Volatility (SV Model_ t distribution)"), size = 1) +
  geom_line(aes(y = RealizedVol, color = "Realized Volatility (21-day Rolling SD)"), size = 0.5) +
  labs(
    title = "Predicted vs Realized Volatility",
    x = "Time",
    y = "Volatility",
    color = "Legend"
  ) +
  theme_minimal()
p1/p2
```

    ## Warning: Removed 20 rows containing missing values or values outside the scale range (`geom_line()`).
    ## Removed 20 rows containing missing values or values outside the scale range (`geom_line()`).

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

3.  **GARCH models**

The same procedure is applied to the Garch models, so that we are able
to extract the conditional volatility and plot it against the 21-day
rolling volatility of the returns. The visual comparison helps us
evaluate how well the models adapt to periods of high and low market
turbulence, and understand if the models are able to catch shocks.

``` r
egarch_vol <- sigma(egarch_fit)
plot(egarch_vol, type="l")
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
min_len <- min(length(egarch_vol), length(realized_vol), length(returns))
df <- data.frame(
  Time = 1:min_len,
  PredictedVol = egarch_vol[1:min_len],
  RealizedVol = realized_vol[1:min_len]
)

p3 <- ggplot(df, aes(x = Time)) +
  geom_line(aes(y = PredictedVol, color = "Predicted Volatility (EGARCH)"), size = 1) +
  geom_line(aes(y = RealizedVol, color = "Realized Volatility (21-day Rolling SD)"), size = 0.5) +
  labs(
    title = "Predicted vs Realized Volatility",
    x = "Time",
    y = "Volatility",
    color = "Legend"
  ) +
  theme_minimal()
gjr_vol<-sigma(gjr_fit)
plot(gjr_vol)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

``` r
min_len <- min(length(gjr_vol), length(realized_vol), length(returns))
df <- data.frame(
  Time = 1:min_len,
  PredictedVol = gjr_vol[1:min_len],
  RealizedVol = realized_vol[1:min_len]
)

p4 <- ggplot(df, aes(x = Time)) +
  geom_line(aes(y = PredictedVol, color = "Predicted Volatility (GJR-GARCH)"), size = 1) +
  geom_line(aes(y = RealizedVol, color = "Realized Volatility (21-day Rolling SD)"), size = 0.5) +
  labs(
    title = "Predicted vs Realized Volatility",
    x = "Time",
    y = "Volatility",
    color = "Legend"
  ) +
  theme_minimal()

garch_vol<-sigma(garch_fit)
plot(garch_vol)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-18-3.png)<!-- -->

``` r
min_len <- min(length(garch_vol), length(realized_vol), length(returns))
df <- data.frame(
  Time = 1:min_len,
  PredictedVol = garch_vol[1:min_len],
  RealizedVol = realized_vol[1:min_len]
)

p5 <- ggplot(df, aes(x = Time)) +
  geom_line(aes(y = PredictedVol, color = "Predicted Volatility (sGARCH)"), size = 1) +
  geom_line(aes(y = RealizedVol, color = "Realized Volatility (21-day Rolling SD)"), size = 0.5) +
  labs(
    title = "Predicted vs Realized Volatility",
    x = "Time",
    y = "Volatility",
    color = "Legend"
  ) +
  theme_minimal()

p3/p4/p5
```

    ## Warning: Removed 20 rows containing missing values or values outside the scale range (`geom_line()`).
    ## Removed 20 rows containing missing values or values outside the scale range (`geom_line()`).
    ## Removed 20 rows containing missing values or values outside the scale range (`geom_line()`).

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-18-4.png)<!-- -->

## **Backtesting the model and analyzing the violations**

In this section, we evaluate the predictive accuracy of the models by
comparing the forecasted volatility against the actual asset returns.
The goal is to determine how well the model captures extreme variations
in the market by calculating the number of violations outside a 90%
confidence interval we build by estimating the 5% and 95% quantiles.

We verifiy whenever the realized returns fall below or exceed the lower
and upper bound of this predictive interval. In these cases, we register
what is called a *violation,* which are crucial to indicate where the
model fails to anticipate extreme movements in the financial series.

1.  **Garch models**

``` r
y <- as.numeric(returns)
y <- as.numeric(y)
egarch_vol <- as.numeric(egarch_vol)
egarch_sum_lower<-sum(y<qnorm(0.05,0,egarch_vol))/length(y)
egarch_sum_upper<-sum(y>qnorm(0.95,0,egarch_vol))/length(y)
lower <- qnorm(0.05, 0, egarch_vol)
upper <- qnorm(0.95, 0, egarch_vol)
viol_lower <- which(y < lower)
viol_upper <- which(y > upper)
plot(y, type = "l", col = "black", lwd = 1,
     ylim = range(c(lower, upper, y)),
     ylab = "Volatility / Returns", xlab = "Time",
     main = "Volatility Forecast with 90% CI and Violations")
lines(lower, col = "orange", lty = 2)
lines(upper, col = "green", lty = 2)
points(viol_lower, y[viol_lower], col = "red", pch = 16)
points(viol_upper, y[viol_upper], col = "blue", pch = 16)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
y <- as.numeric(returns)
y <- as.numeric(y)
garch_vol <- as.numeric(garch_vol)
garch_sum_lower<-sum(y<qnorm(0.05,0,garch_vol))/length(y)
garch_sum_upper<-sum(y>qnorm(0.95,0,garch_vol))/length(y)
lower <- qnorm(0.05, 0, garch_vol)
upper <- qnorm(0.95, 0, garch_vol)
viol_lower <- which(y < lower)
viol_upper <- which(y > upper)
plot(y, type = "l", col = "black", lwd = 2,
     ylim = range(c(lower, upper, y)),  # include tutto nel grafico
     ylab = "Volatility / Returns", xlab = "Time",
     main = "Volatility Forecast with 90% CI and Violations")
lines(lower, col = "orange", lty = 2)
lines(upper, col = "green", lty = 2)
points(viol_lower, y[viol_lower], col = "red", pch = 16)
points(viol_upper, y[viol_upper], col = "blue", pch = 16)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
y <- as.numeric(returns)
y <- as.numeric(y)
gjr_vol <- as.numeric(gjr_vol)
gjr_sum_lower<-sum(y<qnorm(0.05,0,gjr_vol))/length(y)
gjr_sum_upper<-sum(y>qnorm(0.95,0,gjr_vol))/length(y)
lower <- qnorm(0.05, 0, gjr_vol)
upper <- qnorm(0.95, 0, gjr_vol)
viol_lower <- which(y < lower)
viol_upper <- which(y > upper)
plot(y, type = "l", col = "black", lwd = 2,
     ylim = range(c(lower, upper, y)),  # include tutto nel grafico
     ylab = "Volatility / Returns", xlab = "Time",
     main = "Volatility Forecast with 90% CI and Violations")
lines(lower, col = "orange", lty = 2)
lines(upper, col = "green", lty = 2)
points(viol_lower, y[viol_lower], col = "red", pch = 16)
points(viol_upper, y[viol_upper], col = "blue", pch = 16)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

``` r
df_upper_viol <- data.frame(
  EGARCH = egarch_sum_upper,
  GARCH = garch_sum_upper,
  GJR = gjr_sum_upper
)

df_lower_viol <- data.frame(
  EGARCH = egarch_sum_lower,
  GARCH = garch_sum_lower,
  GJR = gjr_sum_lower
)
print(df_lower_viol)
```

    ##       EGARCH      GARCH        GJR
    ## 1 0.05143007 0.05352926 0.05379166

``` r
print(df_upper_viol)
```

    ##       EGARCH      GARCH        GJR
    ## 1 0.03726056 0.03909735 0.03778536

The tables reports the percentage proportion of returns that exceeded
the upper bound and the lower bound of the 90% confidence interval, as
predicted by GARCH models.

For the upper bound:

- The **simple GARCH** model shows the highest upper violation rate at
  3.91%, suggesting it underestimates upper-tail risk slightly less
  conservatively than the others.

- The **GJR-GARCH** and **EGARCH** models have slightly lower violation
  rates, at 3.78% and 3.73%, respectively, indicating a more
  conservative behavior in predicting extreme positive returns.

All models fall below the theoretical 5% threshold, implying that their
predictive intervals are slightly too wide and may be overly cautious in
capturing upper-tail events.

For the lower bound:

- The **EGARCH** model shows a lower-tail violation rate of **5.14%**,
  slightly exceeding the nominal level, suggesting a modest
  underestimation of downside risk.

- Both the **GARCH** (**5.35%**) and **GJR-GARCH** (**5.38%**) models
  exhibit even higher violation rates, implying a greater frequency of
  observed negative returns falling below their predicted lower bounds.

These results indicate that all three models slightly **undercover the
lower tail**, with GARCH and GJR-GARCH models showing a stronger
tendency to underestimate extreme negative movements in returns. This
could point to limited responsiveness to volatility clustering or
asymmetric shocks in downward trends.

2.  **SV models**

``` r
y <- as.numeric(returns)
y <- as.numeric(y)
sv_lev <- as.numeric(sy_lev*exp(h_lev/2))
lev_lower_viol<-sum(y<qnorm(0.05,0,sy_lev*exp(h_lev/2)))/length(y)
lev_upper_viol<-sum(y>qnorm(0.95,0,sy_lev*exp(h_lev/2)))/length(y)
lower <- qnorm(0.05, 0, sy_lev*exp(h_lev/2))
upper <- qnorm(0.95, 0, sy_lev*exp(h_lev/2))
viol_lower <- which(y < lower)
viol_upper <- which(y > upper)
plot(y, type = "l", col = "black", lwd = 2,
     ylim = range(c(lower, upper, y)),  # include tutto nel grafico
     ylab = "Volatility / Returns", xlab = "Time",
     main = "Volatility Forecast with 90% CI and Violations")
lines(lower, col = "orange", lty = 2)
lines(upper, col = "green", lty = 2)
points(viol_lower, y[viol_lower], col = "red", pch = 16)
points(viol_upper, y[viol_upper], col = "blue", pch = 16)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
y <- as.numeric(returns)
y <- as.numeric(y)
sv_lev <- as.numeric(sy_t*exp(h_t/2))
t_lower_viol<-sum(y<qnorm(0.05,0,sy_t*exp(h_t/2)))/length(y)
t_upper_viol<-sum(y>qnorm(0.95,0,sy_t*exp(h_t/2)))/length(y)
lower <- qnorm(0.05, 0, sy_t*exp(h_t/2))
upper <- qnorm(0.95, 0, sy_t*exp(h_t/2))
viol_lower <- which(y < lower)
viol_upper <- which(y > upper)
plot(y, type = "l", col = "black", lwd = 2,
     ylim = range(c(lower, upper, y)),  # include tutto nel grafico
     ylab = "Volatility / Returns", xlab = "Time",
     main = "Volatility Forecast with 90% CI and Violations")
lines(lower, col = "orange", lty = 2)
lines(upper, col = "green", lty = 2)
points(viol_lower, y[viol_lower], col = "red", pch = 16)
points(viol_upper, y[viol_upper], col = "blue", pch = 16)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

``` r
df_upper_viol <- data.frame(
  SV_leverage = lev_upper_viol,
  SV_tdist= t_upper_viol
  
)

df_lower_viol <- data.frame(
  SV_leverage = lev_lower_viol,
  Sv_tdist = t_lower_viol
  
)
print(df_lower_viol)
```

    ##   SV_leverage   Sv_tdist
    ## 1  0.05825243 0.04985568

``` r
print(df_upper_viol)
```

    ##   SV_leverage   SV_tdist
    ## 1  0.04172133 0.04303332

For the lower bound:

- The **SV with leverage** model reports a lower-tail violation rate of
  **5.83%**, exceeding the nominal 5% threshold. This suggests a mild
  **undercoverage** of extreme negative returns, indicating that the
  model underestimates the downside risk slightly.

- In contrast, the **SV with Student’s t distribution** shows a lower
  violation rate of **4.99%**, almost exactly aligned with the expected
  value. This suggests that it provides **well-calibrated coverage** in
  the left tail of the return distribution.

For the upper bound:

- The **SV with leverage** model shows an upper violation rate of
  **4.17%**, slightly below the expected 5%, suggesting a marginal
  **overcoverage**. This implies that the model produces slightly
  conservative upper bounds, overestimating the extent of extreme
  positive returns.

- The **SV with Student’s t-distribution** has an upper violation rate
  of **4.30%**, also below the 5% threshold but slightly higher than the
  leverage model. This indicates a better-calibrated behavior, though
  still conservative in the upper tail.

Both models demonstrate good performance, staying close to the nominal
coverage level. However, their slight overestimation in the upper tail
reflects a tendency to issue wider-than-necessary prediction intervals
for extreme positive returns, ensuring caution in volatile market
scenarios.

On the basis the violation rates observed for both the lower and upper
bounds of the 90% predictive intervals, the stochastic volatility model
with Student’s t-distribution demonstrates the most reliable performance
among the five models considered. Its violation percentages are 4.99% on
the lower tail and 4.30% on the upper tail, both closely aligned with
the theoretical nominal rate of 5%. This indicates that the model
provides well-calibrated predictive intervals without systematically
under- or overestimating tail risks.

## **Split the series into monthly blocks and fit the GEV distribution to the monthly maxima and minima**

### Introduction to Extreme Value Theory and the Block Maxima Approach

Extreme Value Theory is a statistical framework introduced to model the
behavior of rare and extreme events that reside in the tails of a
distribution. These extremes,like the largest daily or monthly gains or
losses in financial markets, are often critical in risk management,
stress testing, and the estimation of tail-dependent risk measures like
Value-at-Risk (VaR).

One of the classical approach to implement the Extreme Value Theory is
the **Block Maxima method**, which consists of dividing a time series
into non-overlapping blocks, monthly in our cases, and extracting the
maximum and minimum observation from each block. This process produces a
new sequence of extremes, denoted:

$$
M_n = \max \{ X_1, X_2, \ldots, X_n \},
$$

where $X_i$ are i.i.d. observations from an unknown distribution $F$.
According to the **Extremal Types Theorem** (Fisher-Tippett-Gnedenko),
under suitable conditions there exist sequences $a_n > 0$ and
$b_n \in \mathbb{R}$ such that:

$$
\frac{M_n - b_n}{a_n} \xrightarrow{d} G(x), \quad \text{as } n \to \infty,
$$

where $G(x)$ is a non-degenerate limit distribution belonging to the
**Generalized Extreme Value (GEV)** family. The cumulative distribution
function of the GEV is given by:

$$
G(x) = \exp \left\{ -\left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right\}, \quad \text{for } 1 + \xi \left( \frac{x - \mu}{\sigma} \right) > 0,
$$

and is characterized by three parameters:

- $\mu$: location parameter, indicating the central tendency of the
  extremes;
- $\sigma > 0$: scale parameter, measuring the dispersion;
- $\xi$: shape parameter, governing the tail behavior and defining to
  whatever of the GEV distribution the series belongs:
  - $\xi > 0$: Fréchet,
  - $\xi = 0$: Gumbel ,
  - $\xi < 0$: Weibull.

In this exercise, we apply the block maxima method to daily log-returns
of the ENI stock from 2010 to 2024. The series is split into monthly
blocks, from which we extract both the **maximum** and the **minimum**
daily return for each month. Two GEV distributions are then fitted
independently to these sequences.

Parameter estimation is performed via **maximum likelihood estimation
(MLE)** (using the “ismev” package) , which ensures asymptotically
efficient and consistent estimators under regularity conditions. The
fitted GEV models provide a description of the behavior of ENI’s return
distribution, enabling a clear distinction between the dynamics of
extreme gains (upside potential) and extreme losses (downside risk). By
modeling the maxima and minima separately, this approach captures the
inherent **asymmetry** in financial return series, where negative shocks
tend to be more abrupt and severe than the positive one.

``` r
year_month <- format(index(returns), "%Y-%m")
monthly_minima <- tapply(-100 * returns, FUN = max, INDEX = year_month)
monthly_maximum <- tapply(100 * returns, FUN = max, INDEX = year_month)
hist(monthly_minima)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
hist(monthly_maximum)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-26-2.png)<!-- -->

``` r
plot(monthly_maximum,
     type = "l",
     col = "darkgreen",
     lwd = 2,
     xlab = "Month",
     ylab = "Maximum returns (%)",
     las = 2,    
     cex.axis = 0.7)  
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-26-3.png)<!-- -->

``` r
plot(monthly_minima,
     type = "l",
     col = "red",
     lwd = 2,
     xlab = "Mese",
     ylab = "Minimum returns (%)",
     las = 2,
     cex.axis = 0.7)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-26-4.png)<!-- -->

After diving the series into block, and estracting the maximum and the
minimum for each month ow we proceed to fit the GEV to our series:

``` r
myfit_min <- gev.fit(as.numeric(monthly_minima))
```

    ## $conv
    ## [1] 0
    ## 
    ## $nllh
    ## [1] 1140.585
    ## 
    ## $mle
    ## [1] 227.957665 102.884123   0.218062
    ## 
    ## $se
    ## [1] 8.56325664 6.84109970 0.05543552

``` r
myfit_max <- gev.fit(as.numeric(monthly_maximum))
```

    ## $conv
    ## [1] 0
    ## 
    ## $nllh
    ## [1] 1113.665
    ## 
    ## $mle
    ## [1] 221.1847652  92.1359163   0.1550719
    ## 
    ## $se
    ## [1] 7.64686195 5.87510954 0.05217124

By analyzing the results of the fit, we know that:

The positive value of the shape parameter $\xi$ = 0.218 indicates that
the distribution of monthly minima belongs to the **Fréchet family,
which is characterized by heavy tails**, implying that extreme negative
returns although rare can be very large in magnitude.

The relatively large **scale parameter** further reflects the
substantial dispersion observed in the left tail of the return
distribution. Standard errors are low relative to parameter values,
suggesting a reliable and stable estimation.

The shape parameter for maxima is also positive, indicating a
**Fréchet** for the right tail too. However a $\xi$ value of 0.155 is
smaller than that estimated for the minima, suggesting a **lighter
tail** for extreme positive returns compared to extreme losses. This
also implies less variability in the magnitude of extreme gains, due to
a lower estimated **scale parameter**. Standard errors are again low,
confirming the robustness of the estimates.

``` r
gev.diag(myfit_min)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

``` r
gev.diag(myfit_max)
```

![](C:/Users/sarap/OneDrive/Documenti/GitHub/Academic-Projects/garch_vs_sv_eni/README_files/figure-gfm/unnamed-chunk-28-2.png)<!-- -->

After fitting the GEV, we assessed the reliability of the results with a
series of disgnostic plots produced by the gev.diag() function.

For the **monthly minima:**

1.  **Probability plot** and **Q-Q plot** jointly indicate a strong
    agreement between the empirical and theoretical quantiles across the
    central part of the distribution, with points clustering around the
    45-degree line. This suggests that the GEV model provides a good
    approximation to the observed empirical distribution of the monthly
    minima. Nonetheless, slight deviations in the extreme lower
    quantiles hint at possible underestimation of the most severe
    losses, a common probkem when working with limited extreme data.

2.  **Return level plot**: the graph shows that return levels exceeding
    1000 (severe losses) are expected roughly once every 1000 periods.
    The empirical points (black circles) align well within the
    confidence bands of the model (blue curves), confirming the model’s
    validity for extrapolating extreme negative returns. This supports
    the notion that the GEV model is reliable in assessing downside
    risk, even for very rare, catastrophic events.

3.  **Density plot**: the fitted GEV density overlays the empirical
    histogram of block minima and captures the bulk of the distribution
    well. The tails show minor discrepancies, which could result from
    residual temporal dependencies or volatility clustering not
    accounted for in the block maxima approach.

Overall, the diagnostic evidence supports the suitability of the GEV
model for characterizing the left tail of ENI’s return distribution over
the 2010-2024 period. The model yields valuable insights into the
frequency and severity of extreme losses and offers a principled way to
extrapolate beyond the historical data.

For the monthly maximum:

1.  **Probability plot**: it suggests a good agreement between the
    empirical and fitted distributions. The data points lie closely
    along the diagonal, indicating that the GEV model well captures the
    empirical cumulative distribution function (CDF). This supports the
    hypothesis that the monthly maxima conform to an extreme value law,
    at least in the central region of the distribution.

2.  **Quantile plot**: most of the observed maxima align well with the
    theoretical quantiles. However, a few extreme observations in the
    upper tail exceed the fitted line, suggesting that the model may
    slightly underestimate the probability of the most extreme positive
    returns. Such behavior is typical when the shape parameter ξ is
    positive but small, implying a light to moderate upper tail.

3.  **Return level plot**: it shows that values over 800 are expected
    approximately once every 100 periods. The GEV fit again captures the
    bulk of the observations, though some of the most extreme maxima lie
    just above the upper confidence bounds. This suggests a slight
    underestimation of extreme gains by the model—a point already noted
    in the quantile and density diagnostics. Nevertheless, the return
    level plot still offers a reasonable extrapolation framework for
    assessing upside risk.

4.  **Density plot**: the histogram of the observed maxima shows a
    concentration of values around the mode of the distribution, with a
    thin right tail. The overlaid GEV density approximates the main body
    of the distribution well, although it appears to underestimate the
    right tail, in line with the earlier observations from the Q-Q and
    return level plots.

Even for the maximum distribution, the fitted GEV model offers a good
first-order approximation for ENI’s returns.


