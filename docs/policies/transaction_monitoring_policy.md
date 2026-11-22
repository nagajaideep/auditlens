# Transaction Monitoring Policy

## 1. Purpose and scope
This policy sets out how the Firm monitors client payment transactions for indicators of fraud, error
and misappropriation during audit and advisory engagements. It applies to every engagement where the
Firm receives a client transaction ledger, including card, wire, cash-out and internal transfer data.

## 2. Risk scoring
Every transaction receives a model risk score between 0 and 1. Scores at or above the alert threshold of
0.80 create an alert. Scores between 0.50 and 0.80 are placed on a watch list and reviewed in a weekly
batch. The alert threshold is reviewed every quarter by the Model Risk Committee and may be changed only
with the committee's written approval.

## 3. Mandatory alert rules
Independently of the model score, the following always create an alert:
- any single transfer of 10,000,000 or more in the ledger currency;
- a transfer that reduces the sending account balance to exactly zero and is followed by a cash-out
  from the receiving account within 24 hours;
- three or more transfers to newly created counterparties by the same sender within one day;
- any payment to a vendor that is not in the approved vendor master file.

## 4. Alert triage and service levels
Alerts are triaged by a first-line analyst within 2 business days. High-severity alerts (score of 0.95 or
above, or any mandatory rule breach above 1,000,000) must be triaged within 1 business day. An alert that
cannot be cleared by the analyst is escalated to the engagement manager. Alerts confirmed as suspicious
are escalated to the engagement partner and the Firm's Forensic team within 5 business days of creation.

## 5. Clearing an alert
An analyst may clear an alert only after obtaining supporting evidence, such as an invoice, contract,
board approval or bank confirmation. The reason for clearing must be recorded in the case management
system. Clearing an alert without evidence is a breach of this policy.

## 6. False positives
The Firm accepts that monitoring produces false positives. Each quarter the analytics team reports the
false-positive rate by transaction type. If the precision of alerts falls below 20% for two consecutive
quarters, the model must be retrained or recalibrated.

## 7. Record keeping
All alerts, triage notes and clearing evidence are retained for seven years from the date the
engagement report is signed.
