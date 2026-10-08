# Elo leave-one-pair-out diagnostics

Each expected rate is estimated without the displayed matchup. Draws count as half a win.

## Largest Elo prediction mismatches

| Row policy | Opponent | Observed | LOO Elo | Residual | z | W-L-D |
|---|---|---:|---:|---:|---:|---|
| strategic_v6 | strategic_v18 | 0.0% | 15.3% | -15.3% | -11.02 | 0-20-0 |
| strategic_v3 | strategic_v19 | 0.0% | 13.3% | -13.3% | -10.50 | 0-20-0 |
| strategic_v6 | strategic_v19 | 0.0% | 12.8% | -12.8% | -10.45 | 0-20-0 |
| strategic_v14 | strategic_v18 | 0.0% | 6.4% | -6.4% | -9.60 | 0-20-0 |
| strategic_v14 | strategic_v19 | 0.0% | 5.3% | -5.3% | -9.27 | 0-20-0 |
| strategic_v8 | strategic_v17 | 0.0% | 16.8% | -16.8% | -9.10 | 0-20-0 |
| strategic_v7 | strategic_v17 | 0.0% | 16.7% | -16.7% | -9.09 | 0-20-0 |
| strategic_v12 | strategic_v17 | 0.0% | 16.7% | -16.7% | -9.09 | 0-20-0 |
| strategic_v9 | strategic_v17 | 0.0% | 16.2% | -16.2% | -9.04 | 0-20-0 |
| strategic_v3 | strategic_v17 | 0.0% | 16.1% | -16.1% | -9.03 | 0-20-0 |
| strategic_v6 | strategic_v17 | 0.0% | 15.5% | -15.5% | -8.97 | 0-20-0 |
| strategic_v13 | strategic_v17 | 0.0% | 14.7% | -14.7% | -8.91 | 0-20-0 |
| strategic_v10 | strategic_v17 | 0.0% | 13.6% | -13.6% | -8.81 | 0-20-0 |
| strategic_v11 | strategic_v17 | 0.0% | 13.4% | -13.4% | -8.75 | 0-20-0 |
| strategic_v15 | strategic_v17 | 0.0% | 11.3% | -11.3% | -8.59 | 0-20-0 |
| strategic_v14 | strategic_v17 | 0.0% | 6.5% | -6.5% | -7.92 | 0-20-0 |
| strategic_v17 | strategic_v18 | 22.5% | 65.4% | -42.9% | -6.96 | 18-62-0 |
| strategic_v1 | strategic_v17 | 0.0% | 2.6% | -2.6% | -6.86 | 0-20-0 |
| strategic_v17 | strategic_v19 | 19.1% | 57.1% | -38.0% | -5.97 | 13-55-0 |
| strategic_v2 | strategic_v5 | 40.5% | 63.7% | -23.2% | -4.40 | 39-58-3 |

## Policy rating uncertainty and Elo misfit

| Policy | Elo | 95% interval | LOO MAE | Large mismatches |
|---|---:|---:|---:|---:|
| strategic_v19 | 1837.2 | 1806.8–1867.5 | 9.1% | 8 |
| strategic_v17 | 1802.0 | 1763.9–1840.0 | 14.8% | 16 |
| strategic_v18 | 1801.9 | 1772.6–1831.2 | 8.3% | 7 |
| strategic_v4 | 1526.7 | 1509.1–1544.2 | 6.5% | 4 |
| strategic_v8 | 1514.2 | 1496.4–1531.9 | 4.3% | 4 |
| strategic_v7 | 1513.2 | 1495.4–1530.9 | 4.4% | 4 |
| strategic_v12 | 1513.2 | 1495.4–1530.9 | 4.4% | 4 |
| strategic_v9 | 1507.4 | 1489.6–1525.1 | 5.6% | 2 |
| strategic_v3 | 1505.4 | 1487.9–1523.0 | 5.9% | 3 |
| strategic_v6 | 1498.0 | 1480.7–1515.4 | 5.2% | 4 |
| strategic_v13 | 1488.2 | 1470.9–1505.5 | 4.3% | 1 |
| strategic_v10 | 1472.9 | 1455.7–1490.1 | 6.4% | 4 |
| strategic_v11 | 1469.5 | 1451.9–1487.0 | 3.8% | 2 |
| strategic_v15 | 1437.3 | 1420.1–1454.6 | 6.7% | 4 |
| strategic_v16 | 1428.5 | 1410.7–1446.2 | 5.0% | 4 |
| strategic_v2 | 1375.7 | 1357.5–1393.9 | 6.3% | 4 |
| strategic_v14 | 1334.5 | 1314.7–1354.4 | 8.9% | 9 |
| strategic_v5 | 1305.1 | 1283.9–1326.2 | 8.3% | 7 |
| strategic_v1 | 1169.2 | 1140.3–1198.2 | 7.1% | 9 |
