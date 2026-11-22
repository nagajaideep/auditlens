-- Exploratory SQL used in the data-quality report. Run: psql $DATABASE_URL -f sql/eda_queries.sql

-- 1. Volume, value and fraud rate by transaction type
SELECT type,
       COUNT(*)                              AS n_tx,
       ROUND(SUM(amount)::numeric, 0)        AS total_amount,
       ROUND(AVG(amount)::numeric, 2)        AS avg_amount,
       SUM(isfraud)                          AS n_fraud,
       ROUND(100.0 * AVG(isfraud), 4)        AS fraud_pct
FROM transactions_raw
GROUP BY type
ORDER BY n_fraud DESC;

-- 2. Fraud by hour of day (step is hours since start)
SELECT (step - 1) % 24 AS hour, COUNT(*) AS n_tx, SUM(isfraud) AS n_fraud,
       ROUND(100.0 * AVG(isfraud), 4) AS fraud_pct
FROM transactions_raw GROUP BY 1 ORDER BY 1;

-- 3. Balance-identity violations: origin balance doesn't reconcile with amount
SELECT type, COUNT(*) AS violations
FROM transactions_raw
WHERE type IN ('TRANSFER','CASH_OUT','PAYMENT','DEBIT')
  AND ABS(oldbalanceorg - amount - newbalanceorig) > 0.01
GROUP BY type;

-- 4. Accounts emptied in a single transaction (classic PaySim fraud signal)
SELECT COUNT(*) AS emptied, SUM(isfraud) AS fraud_among_emptied
FROM transactions_raw
WHERE oldbalanceorg > 0 AND newbalanceorig = 0 AND amount >= oldbalanceorg;

-- 5. Top 20 receiving accounts ("vendors") by inflow, with their fraud exposure
SELECT namedest, COUNT(*) AS n_in, ROUND(SUM(amount)::numeric, 0) AS inflow, SUM(isfraud) AS fraud_in
FROM transactions_raw
WHERE namedest LIKE 'C%'
GROUP BY namedest
ORDER BY inflow DESC
LIMIT 20;

-- 6. Daily spend (input for forecasting), with a 7-day moving average
SELECT day, total, ROUND(AVG(total) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 0) AS ma7
FROM (SELECT (step - 1) / 24 AS day, SUM(amount) AS total FROM transactions_raw GROUP BY 1) d
ORDER BY day;

-- 7. Rule-based baseline: how good is PaySim's own isFlaggedFraud rule?
SELECT SUM(CASE WHEN isflaggedfraud = 1 AND isfraud = 1 THEN 1 ELSE 0 END) AS tp,
       SUM(CASE WHEN isflaggedfraud = 1 AND isfraud = 0 THEN 1 ELSE 0 END) AS fp,
       SUM(CASE WHEN isflaggedfraud = 0 AND isfraud = 1 THEN 1 ELSE 0 END) AS fn
FROM transactions_raw;
