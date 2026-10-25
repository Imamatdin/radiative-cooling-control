# Sweep summary (test: Jan+Jul, cost saving vs tower-only; % of DP optimum)


## phoenix  (DP optimum 16.58%; operator schedule 94.7%; MPC persistence 98.8%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| DDPG | persistence |  | 5 | 13.32 ± 1.04 | 12.39–14.04 | 80.3 | 87.2 |  | 1/5 |
| DDPG | persistence | yes | 10 | 14.09 ± 0.48 | 13.81–14.38 | 85.0 | 89.9 |  | 0/10 |
| SAC | persistence |  | 10 | 13.75 ± 1.34 | 12.91–14.47 | 82.9 | 91.1 |  | 1/10 |
| SAC | persistence | yes | 10 | 13.39 ± 0.75 | 12.92–13.79 | 80.8 | 85.8 |  | 1/10 |
| TD3 | persistence |  | 10 | 13.74 ± 0.95 | 13.13–14.24 | 82.9 | 89.3 |  | 0/10 |
| TD3 | persistence | yes | 10 | 13.83 ± 1.20 | 13.07–14.48 | 83.4 | 92.2 |  | 0/10 |

## houston  (DP optimum 23.97%; operator schedule 96.8%; MPC persistence 99.1%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| DDPG | none |  | 5 | 21.15 ± 0.86 | 20.51–21.83 | 88.2 | 92.9 |  | 2/5 |
| DDPG | perfect |  | 5 | 22.13 ± 0.36 | 21.81–22.34 | 92.3 | 93.4 | 92.5 | 0/5 |
| DDPG | persistence | yes | 10 | 21.91 ± 0.53 | 21.63–22.25 | 91.4 | 96.7 |  | 0/10 |
| SAC | none |  | 10 | 21.38 ± 0.81 | 20.92–21.86 | 89.2 | 94.2 |  | 2/10 |
| SAC | perfect |  | 10 | 21.91 ± 0.76 | 21.44–22.33 | 91.4 | 95.2 | 91.3 | 1/10 |
| SAC | persistence | yes | 10 | 21.68 ± 0.60 | 21.32–22.02 | 90.5 | 93.6 |  | 1/10 |
| TD3 | none |  | 10 | 21.72 ± 0.75 | 21.27–22.15 | 90.6 | 94.8 |  | 1/10 |
| TD3 | perfect |  | 10 | 21.87 ± 0.48 | 21.59–22.15 | 91.2 | 94.6 | 91.2 | 0/10 |
| TD3 | persistence | yes | 10 | 22.17 ± 0.51 | 21.87–22.48 | 92.5 | 95.4 |  | 0/10 |

## seattle  (DP optimum 29.13%; operator schedule 98.4%; MPC persistence 99.2%)

| algo | forecast | DR | n | cost % (mean ± sd) | 95% CI | % opt | best seed % opt | deployed w/ persistence % opt | collapsed |
|---|---|---|---|---|---|---|---|---|---|
| DDPG | persistence |  | 5 | 26.10 ± 0.76 | 25.54–26.73 | 89.6 | 93.4 |  | 0/5 |
| DDPG | persistence | yes | 10 | 25.83 ± 0.72 | 25.41–26.24 | 88.7 | 92.1 |  | 2/10 |
| SAC | persistence |  | 10 | 25.88 ± 0.82 | 25.40–26.36 | 88.9 | 93.1 |  | 2/10 |
| SAC | persistence | yes | 10 | 25.93 ± 0.79 | 25.47–26.39 | 89.0 | 93.0 |  | 3/10 |
| TD3 | persistence |  | 10 | 25.90 ± 0.44 | 25.63–26.14 | 88.9 | 90.9 |  | 1/10 |
| TD3 | persistence | yes | 10 | 25.88 ± 0.69 | 25.48–26.28 | 88.9 | 92.9 |  | 2/10 |
