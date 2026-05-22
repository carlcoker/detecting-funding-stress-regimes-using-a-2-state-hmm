# Detecting Funding Stress Regimes Using a 2-State HMM

This project uses a simple two state Hidden Markov Model to identify latent funding/liquidity stress regimes in US short term funding markets.

Liquidity and funding stress are central to fixed income and securities financing markets, yet they are not directly observable in real time. This project explores whether market stress conditions can be identified probabilistically using a simple two state Hidden Markov Model (HMM).

Using publicly available funding and volatility indicators, the model attempts to classify periods into latent “calm” and “stress” regimes. The project was motivated by an interest in funding markets, collateral dynamics, and the role liquidity conditions play during periods of market stress such as COVID 2020 and the 2023 regional banking instability.

# Key Findings

- The HMM successfully separates the sample into persistent calm and stress regimes.
- Stress regimes are associated with substantially higher implied volatility.
- Major stress periods, including COVID 2020 and the 2023 regional banking instability, coincide with higher probabilities of the stress regime.
- The high persistence of estimated states suggests that liquidity conditions tend to cluster rather than fluctuate randomly through time.

# Limitations

This project is intentionally simplified and should not be interpreted as a production level liquidity or funding stress model. The Hidden Markov Model uses only a small number of publicly available indicators and therefore cannot fully capture the complexity of repo markets, collateral flows, or institutional balance sheet constraints.

In addition, the model assumes only two latent regimes and Gaussian state dynamics, which may oversimplify real market behaviour during periods of extreme stress. The analysis is also sensitive to the chosen input variables and smoothing assumptions.

The objective of the project is not prediction or trading signal generation, but rather to demonstrate how probabilistic regime based modelling can be used to interpret changing liquidity conditions in financial markets.
