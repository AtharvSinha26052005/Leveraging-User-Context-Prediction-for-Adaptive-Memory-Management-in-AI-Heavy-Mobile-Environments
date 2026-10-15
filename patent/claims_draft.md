# Patent Claims — Detailed Draft

## Independent Claims

### Claim 1 — Method
A computer-implemented method for managing application memory in a mobile computing device, the method comprising:

(a) periodically collecting, by a context collection module executing on a processor, a context snapshot comprising: a current foreground application identifier, a sequence of at least three previously-used application identifiers, a current hour of day, a current day of week, a device battery level, a charging status, an available random access memory (RAM) value, a network connectivity type, and an AI task activity indicator;

(b) generating, by a prediction engine comprising a trained gradient-boosted tree classifier, a prediction result comprising: a predicted next application identifier, a confidence score between 0.0 and 1.0, and a ranked list of top-K candidate applications with associated probabilities;

(c) computing, by a cache scoring engine, a cache priority score S(a) for each application 'a' currently resident in a device application cache, the score being a weighted linear combination of six factors:
  - a recency factor R(a) based on elapsed time since last access,
  - a frequency factor F(a) based on normalized historical access count,
  - a prediction factor P(a) derived from the prediction probability output of step (b),
  - an importance factor I(a) based on application category and AI workload status,
  - a temporal factor T(a) based on time-of-day usage patterns, and
  - a memory pressure penalty M based on current RAM utilization ratio;

(d) identifying eviction candidates as applications with the lowest cache priority scores;

(e) conditionally preloading the predicted application into the cache only when all five of the following conditions are simultaneously satisfied:
  (i) the prediction confidence score exceeds a calibrated confidence threshold,
  (ii) available device RAM exceeds a minimum headroom value,
  (iii) device battery level exceeds a minimum battery threshold,
  (iv) a real-time thrashing score computed from a sliding time window is below a thrashing limit, and
  (v) the predicted application is not already resident in the cache; and

(f) executing the eviction and/or preload actions to update the application cache state.

### Claim 2 — System
A mobile computing system comprising:
  a processor;
  a memory storing a plurality of application caches; and
  a non-transitory computer-readable medium storing instructions that, when executed by the processor, cause the system to perform the method of Claim 1.

### Claim 3 — Thrashing Detection
A computer-implemented method for detecting and mitigating memory thrashing in a mobile device, comprising:
  monitoring eviction events, reload events, and cache miss events within a configurable sliding time window;
  computing a thrashing score based on the rate and recurrence of these events;
  transitioning the memory management system from a Normal operating mode to a Safe operating mode when the thrashing score exceeds an entry threshold; and
  suspending preload operations during Safe mode while maintaining eviction operations;
  transitioning back to Normal mode only after the thrashing score remains below an exit threshold for a configurable stability period.

---

## Dependent Claims

### Claim 4
The method of Claim 1, wherein the trained gradient-boosted tree classifier is an XGBoost model trained using multi-class soft probability objective on 19 features comprising 12 raw features and 7 derived features including ram_utilization, is_peak_hour, is_weekend, session_depth, time_since_last_switch, battery_bucket, and prev_app_category.

### Claim 5
The method of Claim 1, wherein the confidence score is post-hoc calibrated using Platt scaling to ensure that the predicted probability corresponds to the empirical frequency of correct predictions at that confidence level.

### Claim 6
The method of Claim 1, wherein the weights of the six-factor scoring formula are adaptively optimized per-user using grid search over a discretized weight space, followed by perturbation-based local refinement.

### Claim 7
The method of Claim 1, wherein AI workload applications — including on-device language model inference engines and computer vision processing frameworks — are assigned elevated importance scores and increased eviction resistance due to their disproportionately large cold-start reload cost.

### Claim 8
The method of Claim 1, further comprising an online learning module that:
  monitors prediction accuracy using a rolling accuracy window;
  detects concept drift in user behavior using KL-divergence and Population Stability Index (PSI) metrics;
  triggers sliding-window model retraining when drift is detected; and
  performs automatic rollback to a previous model version if the retrained model's accuracy is lower.

### Claim 9
The method of Claim 1, wherein the prediction engine further comprises a first-order Markov chain model as a fallback predictor, used when the gradient-boosted tree classifier's confidence score falls below a minimum usable threshold.

### Claim 10
The method of Claim 3, wherein the sliding time window is 120 seconds, the entry threshold is 0.70, the exit threshold is 0.30, and the stability period is 60 seconds.

---

## Notes for Patent Attorney

1. Claims 1-3 are independent; Claims 4-10 are dependent.
2. The key novel element is **Claim 1(c)** — the prediction-integrated cache scoring formula. No prior art uses ML prediction probability as a numerical weight in cache eviction decisions.
3. Claim 5 (confidence calibration) and Claim 8 (online learning with rollback) are additional novel contributions.
4. Prior art search should focus on: US patents related to LMKD, iOS memory management, Google's app prediction patents, and Samsung's memory optimization patents.
