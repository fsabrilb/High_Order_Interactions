# Generalized Shannon Index

[Access this dataset on Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.6q573n65s)

Multifractality is a concept that extends locally the usual ideas of fractality in a system. Nevertheless, the multifractal approaches used lack a multifractal dimension tied to an entropy index like the Shannon index. This paper introduces a generalized Shannon index (GSI) and demonstrates its application in understanding system fluctuations. To this end, traditional multifractality approaches are explained. Then, using the temporal Theil scaling and the diffusive trajectory algorithm, the GSI and its partition function are defined. Next, the multifractal exponent of the GSI is derived from the partition function, establishing a connection between the temporal Theil scaling exponent and the generalized Hurst exponent. Finally, this relationship is verified in a fractional Brownian motion and applied to financial time series. This leads us to propose an approximation denominated *local approximation of fractional Brownian motion (LA-fBm)*, where multifractal systems are viewed as a local superposition of distinct fractional Brownian motions with varying monofractal exponents. Also, we furnish an algorithm for identifying the optimal $q$-th moment of the probability distribution associated with an empirical time series to enhance the accuracy of generalized Hurst exponent estimation.

## File structure

The structure of the data repository consists of:

*   ***Input files:*** It corresponds to the input data sets for the study developed with the generalized Shannon index. There are two types of input files:
    *   **raw_data:** The raw data can correspond to time series data as long as the study variables are defined positively. For instance, the closing price of a stock index, the temperature of a city, the intensity of electric current of a city, etc.
    *   **processed_data:** The processed data corresponds to the processing of different time series data where four types of time series have been calculated: returns, logarithmic returns (*log-returns*), absolute value of the logarithmic returns (*absolute log-returns*), and volatility of logarithmic returns measured through a normalization of the maximum value of the absolute log-returns and the z-score of the log-returns (*volatility of log-returns*).

    It is important to mention that the data of this study does not contain raw data but only processed data. Indeed, only financial time series as currencies or stock indices extracted from YahooFinance are considered. In these time series, processed data is obtained as a data frame where the time series of log-returns, absolute log-returns, and volatility of the log-returns are calculated. Nevertheless, the possibility of including other types of time series data is left for readers interested in this work.

*   ***Modules:*** It corresponds to the different modules developed in Python for the development of the study, namely:
    *   **estimate_hurst_mfdfa.py:** Module designed for the calculation of the generalized Hurst exponent $H(q)$ on different time series using the Multifractal Detrended Fluctuation Analysis (*MF-DFA*) method. It is important to mention that the Hurst exponent is a measure of the long correlation of a time series such that if $H=1/2$, there are no long-range correlations, if $1\geq H>1/2$, the increments of the time series are positively correlated, and if $H<1/2$, the increments of the time series are negatively correlated.

    *   **estimate_temporal_fluctuation_scaling.py:** Module designed to calculate the temporal fluctuation scaling (*TFS*) exponent as data accumulates in the time series. It is important to mention that the TFS is a power law type relationship between the mean ($M_{1}$) and the variance ($\Xi_{2}$) of a time series of the form $\Xi_{2}(t)=K_{TFS}(t)M_{1}^{\alpha_{TFS}(t)}(t)$.

    *   **estimate_temporal_theil_scaling.py:** Module designed to calculate the temporal Theil index scaling (*TTS*) exponent as data accumulates in diffusive trajectories time series. It is important to mention that the TTS is a power law type relationship between the Shannon index normalized to its maximum value ($\mathbb{S}(t)=S(t)/S_{max}$) and the mean of diffusive trajectory normalized to its maximum value ($\mathcal{M}_{1}(t)/\mathcal{M}_{max}$) of a time series of the form $\mathbb{S}(t)=K_{TTS}(t)\left|1-\mathcal{M}_{1}(t)/\mathcal{M}_{max}\right|^{\alpha_{TTS}(t)}(t)$.

    *   **estimate_theil_index.py:** Module designed to calculate the Theil index ($T$), or the generalized entropy index of a set of $N$ data. From these, the Shannon index ($S$) is estimated as the Theil index normalized to its maximum value, that is, $S=\log{(N)}-T$. It is important to mention that Theil index is an inequality measure devised by economist Henri Theil and formulated in terms of an entropy index.

    *   **get_financial_time_series.py:** Module designed to process financial time series extracted from YahooFinance through the ```yfinance``` Python library and its ticker. Once the YahooFinance data is downloaded with this module, other functions estimate the returns with the daily closing prices, the logarithmic returns (*log-returns*), absolute value of the log-returns, and volatility of the log-returns. An example of the metadata output after this processing is mentioned in the following section (see [Metadata of the data sets](#metadata-of-the-data-sets)).

    *   **plot_hurst_tfs.py:** Module designed to graph the final results of the study emphasizing the relationship between the generalized Hurst exponent $H(q)$ as a function of the temporal Theil index scaling exponent $\alpha_{TTS}(t)$. Additionally, other graphs are shown such as the evolution of the generalized Hurst exponent and the temporal Theil scaling exponent over time.

*   ***Logs (optional):*** It corresponds to an optional folder in which different log files are generated to know what is failing in any of the parallelized functions in the different modules of the data repository if any of these files suddenly stops working.

*   ***Output files:*** It corresponds to the folder with the output files after processing different data sets. For example, in this folder, the figures and tables for analysis will be by default. Some of these analyses are to show the estimation of the temporal fluctuation scaling over a time series, the estimation of the temporal Theil index scaling, or the estimation of the generalized Hurst exponent of a time series using the Multifractal Detrended Fluctuation Analysis (*MF-DFA*) method.

*   ***Scripts:*** It corresponds to different Jupyter notebooks where the study analyses were carried out and to emphasize some additional aspects, a section is dedicated to them later.

## Metadata of the data sets

The metadata of the different data sets that appear in this repository are organized by the ```.csv``` or ```.xlsx``` files placed in the input_files and output_files folders, namely:

|       **Folder**      |                  **Data set**                 |   **Dataset type**   |
|:---------------------:|:---------------------------------------------:|:--------------------:|
|      input_files      |            df_currency_20230608.csv           |        df_fts        |
|      input_files      |         df_hurst_currency_20230608.csv        |       df_hurst       |
|      input_files      |          df_tfs_currency_20230608.csv         |        df_tfs        |
|      input_files      |          df_tts_currency_20230608.csv         |        df_tts        |
|      input_files      |          df_stock_index_20230608.csv          |        df_fts        |
|      input_files      |       df_hurst_stock_index_20230608.csv       |       df_hurst       |
|      input_files      |        df_tfs_stock_index_20230608.csv        |        df_tfs        |
|      input_files      |        df_tts_stock_index_20230608.csv        |        df_tts        |
|      output_files     |       df_hurst_tts_currency_20230608.csv      |     df_hurst_tts     |
|      output_files     |   df_hurst_tts_optimal_currency_20230608.csv  | df_hurst_tts_optimal |
|      output_files     |     df_hurst_tts_stock_index_20230608.csv     |     df_hurst_tts     |
|      output_files     | df_hurst_tts_optimal_stock_index_20230608.csv | df_hurst_tts_optimal |
| output_files/20230701 |           df_hurst_tts_20230701.xlsx          |   df_hurst_tts_fbm   |

The ```Dataset type``` column indicates the type of table that is available since some of these schemes are repeated between the different data sets considered (stock indices and currencies). Thus, the types of data sets available are:

*   ```df_fts```: Corresponds to the data set of multiple financial time series (*fts*) data processed and hosted as processed data.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | date | datetime | Dates in ISO 8601 format ("YYYY-mm-dd") | Date on which the data was extracted from YahooFinance |
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | ticker_name | string | Alphanumeric character string | Name assigned to the YahooFinance ticker |
    | step | int | Positive integer values | Number of days since the minimum or initial date that data is downloaded from YahooFinance |
    | Open | float | Positive numeric values | Price opening value of a financial time series on a specific date |
    | High | float | Positive numeric values | Highest price value of a financial time series on a specific date |
    | Low | float | Positive numeric values | Lowest price value of a financial time series on a specific date |
    | Close | float | Positive numeric values | Closing value of the price of a financial time series on a specific date |
    | Adj Close | float | Positive numeric values | Adjusted closing value of the price of a financial time series on a specific date |
    | Volume | float | Positive numeric values | Price transaction volume of a financial time series on a specific date |
    | return | float | Numeric values | Closing price returns of a financial time series between two consecutive dates |
    | log_return | float | Numeric values | Closing price logarithmic returns of a financial time series between two consecutive dates |
    | absolute_log_return | float | Positive numeric values | Absolute value of the logarithmic returns of a financial time series between two consecutive dates |
    | log_volatility | float | Positive numeric values | Volatilities of the logarithmic returns of a financial time series between two consecutive dates |
    | cum_log_return | float | Numeric values | Cumulative logarithmic returns since the initial date |
    | cum_absolute_log_return | float | Positive numeric values | Cumulative absolute value of the logarithmic returns since the initial date |
    | cum_log_volatility | float | Positive numeric values | Cumulative volatilities of the logarithmic returns since the initial date |
    | cummean_log_return | float | Numeric values | Cumulative mean of the logarithmic returns since the initial date |
    | cummean_absolute_log_return | float | Positive numeric values | Cumulative mean of the absolute value of the logarithmic returns since the initial date |
    | cummean_log_volatility | float | Positive numeric values | Cumulative mean of the volatilities of the logarithmic returns since the initial date |
    | cumvariance_log_return | float | Positive numeric values | Cumulative variance of the logarithmic returns since the initial date |
    | cumvariance_absolute_log_return | float | Positive numeric values | Cumulative variance of the absolute value of the logarithmic returns since the initial date |
    | cumvariance_log_volatility | float | Positive numeric values | Cumulative variance of the volatilities of the logarithmic returns since the initial date |

*   ```df_hurst```: Corresponds to the data set obtained after the estimation of the generalized Hurst exponent $H(q)$ using the Multifractal Detrended Fluctuation Analysis (*MF-DFA*) method for different orders in the moments ($q$).

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | q_order | float | Numeric values | qth order used to estimate the generalized Hurst exponent |
    | dfa_degree | int | Positive integer values | Detrended fluctuation analysis order |
    | hurst | float | Numeric values | Value of the generalized Hurst exponent |
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | step | int | Positive integer values | Number of days since the minimum or initial date that data is downloaded from YahooFinance |
    | time_series | string | Alphabetic character string | Type of time series analyzed (log-return, absolute log-return or log-return volatility) |

*   ```df_tfs``` Corresponds to the data set obtained after adjusting the temporal fluctuation scaling (*TFS*) as a power law between the mean and the accumulated variance in a time series.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | max_step | int | Positive integer values | Number of days since the minimum or initial date that data is downloaded from YahooFinance |
    | time_series | string | Alphabetic character string | Type of time series analyzed (log-return, absolute log-return or log-return volatility) |
    | p_norm | float | Numeric values | pth norm used to estimate the mean average error of order p |
    | coefficient_tfs | float | Positive numeric values | Estimated temporal fluctuation scaling coefficient |
    | error_coefficient_tfs | float | Positive numeric values | Error in the estimated temporal fluctuation scaling coefficient |
    | exponent_tfs | float | Numeric values | Estimated temporal fluctuation scaling exponent |
    | error_exponent_tfs | float | Positive numeric values | Error in the estimated temporal fluctuation scaling exponent |
    | average_error_tfs | float | Positive numeric values | Mean average error of the temporal fluctuation scaling fitting as power-law relation |
    | rsquared_tfs | float | Positive numeric values | Coefficient of determination of the temporal fluctuation scaling fitting as power-law relation |

*   ```df_tts``` Corresponds to the data set obtained after adjusting the temporal Theil index scaling (*TTS*) as a power law between the Shannon index normalized to its maximum value and the mean of diffusive trajectory normalized to its maximum value.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | max_step | int | Positive integer values | Number of days since the minimum or initial date that data is downloaded from YahooFinance |
    | time_series | string | Alphabetic character string | Type of time series analyzed (log-return, absolute log-return or log-return volatility) |
    | p_norm | float | Numeric values | pth norm used to estimate the mean average error of order p |
    | coefficient_tts | float | Positive numeric values | Estimated temporal Theil index scaling coefficient |
    | error_coefficient_tts | float | Positive numeric values | Error in the estimated temporal Theil index scaling coefficient |
    | exponent_tts | float | Numeric values | Estimated temporal Theil index scaling exponent |
    | error_exponent_tts | float | Positive numeric values | Error in the estimated temporal Theil index scaling exponent |
    | average_error_tts | float | Positive numeric values | Mean average error of the temporal Theil index scaling fitting as power-law relation |
    | rsquared_tts | float | Positive numeric values | Coefficient of determination of the temporal Theil index scaling fitting as power-law relation |

*   ```df_hurst_tts``` Corresponds to the data set obtained after adjusting the generalized Hurst exponent $H(q)$ based on the temporal Theil scaling exponent $\alpha_{TTS}(t)$, as a polynomial relationship (*local approximation of fractional Brownian motion (LA-fBm)*).

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | time_series | string | Alphabetic character string | Type of time series analyzed (log-return, absolute log-return or log-return volatility) |
    | q_order | float | Numeric values | qth order used to estimate the generalized Hurst exponent |
    | parameters_tts | float | Numeric values | Estimated coefficients of the polynomial relation between generalized Hurst exponent and the temporal Theil index scaling exponent |
    | error_parameters_tts | float | Positive numeric values | Error in the estimated coefficients of the polynomial relation between generalized Hurst exponent and the temporal Theil index scaling exponent |
    | rsquared_tts | float | Positive numeric values | Mean average error of the polynomial fitting between generalized Hurst exponent and the temporal Theil index scaling exponent |
    | average_error_tts | float | Positive numeric values | Coefficient of determination of the polynomial fitting between generalized Hurst exponent and the temporal Theil index scaling exponent |

*   ```df_hurst_tts_optimal``` Corresponds to the data set obtained after performing multiple adjustments between the generalized Hurst exponent $H(q)$ as a function of the temporal Theil scaling exponent $\alpha_{TTS}(t)$, for different values of moments $q$. Then, by constructing a matrix with the adjustment coefficients, the matrix norm of said matrix and its associated eigenvectors are estimated.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | time_series | string | Alphabetic character string | Type of time series analyzed (log-return, absolute log-return or log-return volatility) |
    | q_order | float | Numeric values | qth order used to estimate the generalized Hurst exponent |
    | matrix_norm | float | Positive numeric values | Matrix norm of the matrix constructed with the adjustment coefficients after performing multiple regressions between $H(q)$ and $\alpha_{TTS}(t)$ for different values of moments $q$ |
    | eigenvector | float | Numeric values | Eigenvector of the matrix constructed with the adjustment coefficients after performing multiple regressions between $H(q)$ and $\alpha_{TTS}(t)$ for different values of moments $q$ |

*   ```df_hurst_tts_fbm``` Corresponds to the data set obtained after doing multiple simulations of fractional Brownian motion (*fBm*) and estimating the linear relationship that should exist between the Hurst exponent ($H$) and the temporal Theil index scaling exponent $\alpha_{TTS}(t)$.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | time_series | string | Alphabetic character string | Type of time series analyzed for fractional Brownian motion (original data, absolute log-return or log-return volatility) |
    | n_simulation | int | Positive integer values | Numerical identifier of the simulation of fractional Brownian motion |
    | hurst_exponent | float | Numeric values | Generalized Hurst exponent estimated with the Multifractal Detrended Fluctuation Analysis (MF-DFA) method |
    | hurst_exponent_sd | float | Positive numeric values | Standard deviation of the generalized Hurst exponent estimated over simulations |
    | tts_exponent | float | Numeric values | Estimated temporal Theil index scaling exponent per simulation |
    | tts_sd | float | Positive numeric values | Standard deviation of the estimated temporal Theil index scaling exponent over simulations |
    | real_hurst_exponent_sd | float | Numeric values | Normalized standard deviation of the generalized Hurst exponent using the square root of the number of simulations |
    | real_tts_exponent_sd | float | Positive numeric values | Normalized standard deviation of the temporal Theil index scaling exponent using the square root of the number of simulations |

## Scripts order

The set of codes developed for this data repository is divided into two parts specified below.

### Fractional Brownian motion (fBm)

To estimate a relationship between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$, multiple simulations of fractional Brownian motion (*fBm*) are generated to which the Multifractal Detrended Fluctuation Analysis (*MF-DFA*) method is applied to verify the value of the Hurst exponent. Furthermore, the diffusive trajectory algorithm is applied to estimate the temporal Theil scaling exponent $\alpha_{TTS}(t)$ of an fBm. Finally, the linear relationship that exists between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$ is verified. To do this, you have the following scripts:

1.  estimate_hurst_tts_relation_fbm
2.  plot_hurst_tts_relation

### Financial time series (fts)

To show the relationship between the temporal Theil scaling exponent $\alpha_{TTS}(t)$ and the generalized Hurst exponent $H(q)$ in an arbitrary empirical time series, the Dow Jones and Euro to Colombian peso financial time series are used as an example. From these results, a method is implemented to calculate the most optimal value of the $q$-th moment within a set of test values $q_{1}$, $q_{2}$, ..., $q_{W}$. To do this, you have the following scripts:

1.  process_data_fts
2.  estimate_hurst_exponent_mfdfa_fts
3.  estimate_tfs_fts
4.  estimate_diffusive_algorithm_fts
5.  estimate_tts_fts
6.  prepare_exponents_data
7.  compare_exponents

## Code/Software

All the information shown in this data repository is organized in the different folders mentioned and with all the files shown in the following public Github repository [[1]](#references).

To run the different notebooks in the ```scripts``` folder, it is recommended to use version 2.1.4 of ```pandas``` and version 1.24.4 of ```numpy```. Also, it is recommended to install other Python libraries such as ```yfinance```, ```MFDFA``` and ```tqdm```.

## References

\[1] F. Abril. *Generalized Shannon Index*. Github repository. Available on: [https://github.com/fsabrilb/Generalized_Shannon_Index](https://github.com/fsabrilb/Generalized_Shannon_Index)
