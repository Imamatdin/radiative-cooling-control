# Sweep summary (test: Jan+Jul, cost saving vs tower-only; % of DP optimum)


## phoenix  (DP optimum 16.58%; operator schedule 94.7%; MPC persistence 98.8%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| SAC | persistence |  | 6 | 13.76 ± 1.38 | 12.65–14.66 | 83.0 | 90.9 |  | 0/6 |

## houston  (DP optimum 23.97%; operator schedule 96.8%; MPC persistence 99.1%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| SAC | none |  | 6 | 21.25 ± 0.66 | 20.79–21.74 | 88.6 | 93.0 |  | 1/6 |
| SAC | perfect |  | 6 | 22.03 ± 0.90 | 21.34–22.61 | 91.9 | 95.2 | 91.7 | 1/6 |

## seattle  (DP optimum 29.13%; operator schedule 98.4%; MPC persistence 99.2%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| SAC | persistence |  | 6 | 26.09 ± 0.77 | 25.53–26.64 | 89.6 | 93.1 |  | 1/6 |
