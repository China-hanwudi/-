# Temporal N3 Resampling Protocol

1. Restrict candidates to strict-past turns in the same dialogue and freeze a manifest before training.
2. Partition the available history into older 50%, middle 30%, and recent 20% chronological bands. Record the chosen recent-heavy sampling policy, seed, pool percentage/ceiling, and realized quotas in the manifest.
3. Train each authorized round with fresh model parameters and a fresh optimizer. A round cannot read a previous checkpoint, optimizer state, prediction, loss, or metric.
4. Retain candidates using train/development-OOF utility-risk outputs. Refill only from never-selected history and record all retained and refilled identifiers.
5. A failed final gate may authorize no more than two further resampling attempts. Each authorization requires every frozen condition: Macro-F1 gain >= 0.8 pp, Weighted-F1 gain >= 0.5 pp, lower NLL, harm reduction >= 2.0 pp, and improvement in >= 80% of seeds.
6. After the third round, or after any failed authorization condition, route to `current-only`.
7. Test data is inaccessible to sampling, selection, authorization, threshold choice, and training. The one-time test is legal only after the entire selector and receipt chain is frozen.
