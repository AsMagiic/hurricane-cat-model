# Hurricane Cat Model — Florida Homeowners

## Project goal
Build a compound Poisson loss simulation to generate an EP curve 
and read PML metrics. For portfolio into the cat modeling industry.

## Model parameters
- Peril: Atlantic hurricane, Florida coastal homeowners
- TIV: USD 500,000,000
- Frequency: N ~ Poisson(λ=0.7) events/year
- Severity: Lognormal(μ=15.9342, σ=1.0857) in USD
- Simulation: M=100,000 years, seed=42
- Key validation: AAL should converge to λ × E[X] = USD 10.5M/year

## Output targets
- AEP curve (aggregate exceedance probability)
- OEP curve (occurrence exceedance probability)  
- PML at 1-in-100 and 1-in-250 return periods
- AAL

## Stack
Python, numpy, pandas, matplotlib