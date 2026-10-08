# Elo leave-one-pair-out diagnostics

Each expected rate is estimated without the displayed matchup. Draws count as half a win.

## Largest Elo prediction mismatches

| Row policy | Opponent | Observed | LOO Elo | Residual | z | W-L-D |
|---|---|---:|---:|---:|---:|---|
| strategic_v15 | strategic_v17 | 0.0% | 8.6% | -8.6% | -20.87 | 0-20-0 |
| strategic_v14 | strategic_v17 | 0.0% | 5.1% | -5.1% | -19.63 | 0-20-0 |
| strategic_v14 | strategic_v18 | 0.0% | 4.2% | -4.2% | -19.15 | 0-20-0 |
| strategic_v6 | strategic_v19 | 0.0% | 4.5% | -4.5% | -17.71 | 0-20-0 |
| strategic_v17 | strategic_v18 | 20.7% | 56.9% | -36.2% | -17.58 | 141-541-0 |
| strategic_v1 | strategic_v17 | 0.0% | 2.0% | -2.0% | -17.43 | 0-20-0 |
| strategic_v13 | strategic_v17 | 0.5% | 12.1% | -11.6% | -17.01 | 1-288-1 |
| strategic_v14 | strategic_v19 | 0.0% | 1.9% | -1.9% | -16.66 | 0-20-0 |
| strategic_v6 | strategic_v17 | 0.6% | 12.5% | -11.9% | -16.56 | 2-328-0 |
| strategic_v17 | strategic_v19 | 15.8% | 41.7% | -25.9% | -13.49 | 214-1140-0 |
| strategic_v1 | strategic_v14 | 40.4% | 20.0% | +20.4% | +10.76 | 352-521-7 |
| strategic_v1 | strategic_v5 | 16.5% | 35.3% | -18.8% | -10.54 | 146-749-5 |
| strategic_v8 | strategic_v17 | 4.4% | 14.7% | -10.3% | -8.85 | 20-441-1 |
| strategic_v12 | strategic_v17 | 4.5% | 14.8% | -10.3% | -8.83 | 21-447-0 |
| strategic_v9 | strategic_v17 | 3.9% | 13.7% | -9.9% | -8.79 | 17-419-0 |
| strategic_v3 | strategic_v17 | 2.7% | 11.8% | -9.1% | -8.05 | 7-257-0 |
| strategic_v11 | strategic_v17 | 2.1% | 11.0% | -9.0% | -7.65 | 3-166-1 |
| strategic_v2 | strategic_v5 | 35.9% | 57.5% | -21.6% | -7.61 | 120-216-4 |
| strategic_v14 | strategic_v15 | 54.6% | 33.9% | +20.7% | +6.58 | 153-127-2 |
| strategic_v4 | strategic_v17 | 6.3% | 14.1% | -7.8% | -5.91 | 27-410-1 |

## Policy rating uncertainty and Elo misfit

| Policy | Elo | 95% interval | LOO MAE | Large mismatches |
|---|---:|---:|---:|---:|
| strategic_v19 | 2006.4 | 1990.7–2022.1 | 9.0% | 13 |
| strategic_v18 | 1866.8 | 1853.8–1879.9 | 7.3% | 11 |
| strategic_v17 | 1829.3 | 1816.6–1842.0 | 10.0% | 15 |
| strategic_v7 | 1509.6 | 1499.2–1519.9 | 3.8% | 5 |
| strategic_v12 | 1504.6 | 1494.2–1514.9 | 4.1% | 4 |
| strategic_v8 | 1503.8 | 1493.5–1514.2 | 3.7% | 5 |
| strategic_v4 | 1500.6 | 1490.2–1510.9 | 3.2% | 3 |
| strategic_v9 | 1491.8 | 1481.4–1502.2 | 4.7% | 5 |
| strategic_v6 | 1474.2 | 1463.7–1484.7 | 3.7% | 4 |
| strategic_v13 | 1469.6 | 1459.2–1480.1 | 3.5% | 2 |
| strategic_v3 | 1468.4 | 1457.9–1478.8 | 2.6% | 2 |
| strategic_v10 | 1465.2 | 1454.7–1475.6 | 4.6% | 5 |
| strategic_v11 | 1459.5 | 1449.1–1470.0 | 3.0% | 2 |
| strategic_v15 | 1418.3 | 1407.8–1428.8 | 6.7% | 7 |
| strategic_v16 | 1403.9 | 1393.4–1414.4 | 4.6% | 6 |
| strategic_v2 | 1341.7 | 1331.0–1352.4 | 5.7% | 5 |
| strategic_v14 | 1322.5 | 1311.7–1333.3 | 6.8% | 10 |
| strategic_v5 | 1313.8 | 1302.9–1324.7 | 8.0% | 9 |
| strategic_v1 | 1150.0 | 1137.0–1163.0 | 7.9% | 11 |
