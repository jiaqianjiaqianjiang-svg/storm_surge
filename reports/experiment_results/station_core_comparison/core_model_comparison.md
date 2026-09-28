# Aligned core-model comparison

Skill score is `1 - model MSE / persistence MSE` at the same station and lead.

## 1 h

**RMSE (cm)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 1.0023 | 0.8203 | 0.8602 | 0.8791 | 0.8735 |
| Xiamen | 12.0437 | 4.7148 | 3.9245 | 3.4842 | 3.4614 |

**RRMSE (%)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 16.7644 | 13.7209 | 14.3890 | 14.7051 | 14.6107 |
| Xiamen | 75.0158 | 29.3668 | 24.4440 | 21.7020 | 21.5595 |

**Pearson r**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.9909 | 0.9939 | 0.9935 | 0.9932 | 0.9932 |
| Xiamen | 0.8291 | 0.9735 | 0.9817 | 0.9858 | 0.9858 |

**Skill vs persistence**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.0000 | 0.3301 | 0.2633 | 0.2306 | 0.2404 |
| Xiamen | 0.0000 | 0.8467 | 0.8938 | 0.9163 | 0.9174 |


## 24 h

**RMSE (cm)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 2.4298 | 2.1850 | 2.6221 | 2.5999 | 2.4584 |
| Xiamen | 18.0601 | 10.5271 | 8.7182 | 7.9315 | 7.1059 |

**RRMSE (%)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 40.2794 | 36.2214 | 43.4680 | 43.0996 | 40.7541 |
| Xiamen | 112.7042 | 65.6943 | 54.4058 | 49.4965 | 44.3442 |

**Pearson r**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.9469 | 0.9564 | 0.9454 | 0.9468 | 0.9465 |
| Xiamen | 0.6151 | 0.8626 | 0.9108 | 0.9287 | 0.9395 |

**Skill vs persistence**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.0000 | 0.1913 | -0.1646 | -0.1449 | -0.0237 |
| Xiamen | 0.0000 | 0.6602 | 0.7670 | 0.8071 | 0.8452 |


## 72 h

**RMSE (cm)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 3.6790 | 2.9971 | 4.2050 | 4.1700 | 3.7010 |
| Xiamen | 26.8693 | 14.5100 | 13.7232 | 12.1391 | 10.0030 |

**RRMSE (%)**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 60.0957 | 48.9571 | 68.6876 | 68.1150 | 60.4538 |
| Xiamen | 168.5017 | 90.9942 | 86.0602 | 76.1260 | 62.7307 |

**Pearson r**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.8802 | 0.9203 | 0.8804 | 0.8879 | 0.8795 |
| Xiamen | 0.1438 | 0.7289 | 0.7777 | 0.8441 | 0.8804 |

**Skill vs persistence**

| Station | Persistence | Ridge | Fusion CNN | CNN-GRU | Rollout-6 |
| --- | --- | --- | --- | --- | --- |
| Prickly Bay | 0.0000 | 0.3363 | -0.3064 | -0.2847 | -0.0120 |
| Xiamen | 0.0000 | 0.7084 | 0.7391 | 0.7959 | 0.8614 |
