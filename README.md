# Relax Inc. User Adoption Challenge

## Objective
Identify correlates of user adoption and assess whether signup-known variables predict adoption among future users. Adopted means at least three distinct login dates in any inclusive seven-calendar-day period.

## Files
- `analysis.py`: reproducible cleaning, rolling-window labeling, descriptive analysis, 14-day prediction, chronological holdout, and plot.
- `findings.pdf`: one-page executive report.
- `results.txt`: console output with model performance and coefficient estimates.
- `adoption_by_source.csv`, `adoption_labels.csv`, `model_coefficients.csv`, `adoption_by_source.png`: generated results.
- `data/`: provided input CSVs.

## Run
```bash
pip install -r requirements.txt
python analysis.py
```

## Methodological notes
- Duplicate login events on the same calendar date count once.
- 'Ever adopted' is a retrospective descriptive outcome, not a forecast.
- For a forward-looking test, restrict to users with >=14 days of observation and count qualifying adoption within 14 days of signup; use an 80/20 chronological split.
- No post-signup activity or last-session timestamp enters the signup-time predictor.
- Marketing flags and inviter fields may have been updated after signup; predictive interpretation assumes they were available at signup, which cannot be verified from this snapshot.
- Low positive rate and weak holdout performance limit confidence in predictive use.
