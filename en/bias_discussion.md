---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
<div style="float: right;">
  <a href="../de/bias_discussion.html" style="margin-left: 10px;">Deutsch</a>
  <a href="../en/bias_discussion.html">English</a>
</div>

# 4. Discussion and Quiz

## What We Have Seen: A Recap

The previous three chapters each demonstrated a different way that a model can go
wrong not because of a programming error, but because of a flaw in the data or
the analysis process. The table below summarises the four bias types covered.

| Bias | Chapter | What goes wrong | Signature symptom |
|---|---|---|---|
| **Omitted variable** | 1 | An important predictor is left out of the model | Systematic over- or under-prediction for specific groups |
| **Sampling** | 2 | Training data does not represent the target population | Large errors for underrepresented groups at test time |
| **Survivorship** | 3 | Only "successful" cases appear in the data; failures are invisible | Model performs well on survivors but poorly on the general population |
| **P-hacking** | 3 | Tests are repeated until a significant result appears | Published findings that cannot be replicated |

Although they arise in different ways, all four share a common root: the data or the analysis does not faithfully represent the world the model is supposed to describe.

---

## Real-World Implications

### Omitted Variable Bias

In the first chapter, omitting *sex* from a wage model caused the model to
over-predict wages for women and under-predict for men by roughly $5/hr. The
same mechanism appears whenever a relevant but inconvenient variable is excluded.

Real-world examples:
- **Pay equity audits** that control for job title and tenure but omit the fact
  that women are concentrated in lower-paid job families, masking a structural gap.
- **Credit-scoring models** that exclude race (correctly, as required by law) but
  include zip code or education, which correlate strongly with race and reintroduce
  the bias through a back door.
- **Medical risk scores** that were trained on cost-of-care rather than disease
  severity; because Black patients historically received less care, the model
  systematically underestimated their medical needs.

### Sampling Bias

In the second chapter, training only on young workers meant the model never
learned that wages grow steeply with experience. Applied to a 50-year-old, it
predicted a young-worker wage.

Real-world examples:
- **Facial recognition** systems trained predominantly on light-skinned faces that
  perform poorly on darker skin tones. sometimes with false-positive rates ten
  times higher.
- **Clinical trial results** derived from predominantly male or Western populations
  that are then applied to groups not represented in the original study.
- **Recommender systems** built on data from power users (who rate many items)
  that serve casual users poorly.

### Survivorship Bias

In the third chapter, an HR archive was missing the short-tenure, low-wage workers
who had already left: precisely the group hardest to predict. A model trained on
the archive performed well on long-term employees but poorly on new hires.

Real-world examples:
- **Entrepreneurship research** based on currently operating companies ignores the
  majority of start-ups that failed, producing over-optimistic conclusions about
  what makes businesses succeed.
- **Investment strategies** derived from funds that still exist today, ignoring
  the many funds that closed after poor performance: a well-known problem in
  backtesting.
- **Historical salary benchmarks** built from employees who stayed at a company,
  excluding those who left for better pay elsewhere and therefore underestimating
  the competitive market rate.

### P-hacking

In the third chapter, running t-tests on 20 completely random yes/no features
produced one or two "significant" results by chance alone. If only those were
reported, the literature would contain a false claim about wages.

Real-world examples:
- **Drug trials** that test many dosages, sub-populations, or endpoints and report
  only the ones that crossed the significance threshold, contributing to a
  replication crisis in medicine and psychology.
- **Economic policy analyses** where researchers test many model specifications
  (different control variables, sample restrictions, time windows) and report the
  specification that best supports a preferred conclusion.
- **A/B testing** in product development, where running many simultaneous tests
  and acting on whichever one first reaches *p* < 0.05 inflates the false-positive
  rate far above the nominal 5%.

---

## How to Detect and Mitigate Each Bias

### Detection

| Bias | Key diagnostic                                                                                                                                          |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Omitted variable** | Check whether residuals are systematic across a group you did *not* include as a feature                                                                |
| **Sampling** | Compare the distribution of key variables in training data against the target population; evaluate RMSE or accuracy by subgroup                         |
| **Survivorship** | Ask: *who is not in this dataset, and why?* Look for implausibly low failure or dropout rates                                                           |
| **P-hacking** | Check whether all tests and all model specifications were reported; plot the full distribution of p-values. It should be roughly uniform under the null |

### Mitigation

| Bias | Practical remedy |
|---|---|
| **Omitted variable** | Include all theoretically relevant variables; use residual plots to check for remaining group-level patterns |
| **Sampling** | Collect representative data; if impossible, document the gap and evaluate models on held-out data from underrepresented groups |
| **Survivorship** | Seek out data on non-survivors (closed funds, failed firms, drop-outs); weight or flag surviving cases explicitly |
| **P-hacking** | Pre-register hypotheses before collecting data; apply corrections for multiple comparisons (e.g. Bonferroni); report all tests, not just significant ones |

---

## Ethical Considerations

The biases discussed in this tutorial are not abstract statistical curiosities.
They affect real people. A wage model that omits sex may be used to set pay.
A credit model trained on unrepresentative data may deny loans to qualified
applicants. A published p-hacked study may inform policy.

A few principles to keep in mind:

- **Transparency**: Clearly document what variables the model uses, what data it
  was trained on, and what groups are underrepresented. Users of a model cannot
  correct for biases they do not know about.
- **Accountability**: Establish who is responsible for monitoring model performance
  over time. Bias can emerge or shift as the world changes, even if the model
  itself does not.
- **Evaluate by group, not just overall**: Overall accuracy or RMSE can look
  acceptable while hiding severe disparities for specific subgroups. Always break
  down performance metrics by the groups your model will affect.
- **Report all analyses**: Whether in academia or industry, reporting only the
  analysis that gives the most attractive result is a form of p-hacking. A
  transparent workflow documents what was tried and what was found, including
  null results.

---

## Exercises

### Exercise 1: Omitted Variable Bias
```{raw} html
<style>
  .quiz-container {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 2rem;
    max-width: 750px;
    font-family: "Segoe UI", Roboto, sans-serif;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
  }
  .quiz-container p.question {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
  }
  .quiz-container label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.75rem 0;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background-color: #eef2ff;
    cursor: pointer;
  }
  .quiz-container button {
    margin-top: 1.5rem;
    background-color: #4f46e5;
    color: white;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
  }
  .quiz-feedback {
    margin-top: 1.25rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 500;
  }
  .success { background-color: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }
  .error { background-color: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }
  .warning { background-color: #fef9c3; color: #a16207; border: 1px solid #facc15; }
</style>

<div class="quiz-container" id="quiz1">
  <p class="question">In Chapter 1 we built a wage model that omitted the variable <em>sex</em>. What was the main consequence?</p>
  <form id="quiz1-form">
    <label><input type="radio" name="q1" value="a"> The model's overall RMSE became zero</label>
    <label><input type="radio" name="q1" value="b"> The model systematically over-predicted wages for women and under-predicted for men</label>
    <label><input type="radio" name="q1" value="c"> The model refused to make any predictions</label>
    <label><input type="radio" name="q1" value="d"> The model became more accurate because it had fewer inputs to handle</label>
    <button type="button" onclick="checkQ1()">Submit Answer</button>
    <p id="quiz1-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ1() {
  const answers = {
    a: { correct: false, feedback: "Incorrect. Omitting a variable increases error, it does not eliminate it." },
    b: { correct: true,  feedback: "Correct! Without sex in the model, it applied a single average formula to everyone. Because men earn more on average in the data, the model over-predicted for women and under-predicted for men by about $5/hr in each direction." },
    c: { correct: false, feedback: "Incorrect. The model still ran, but its predictions were systematically biased for each group." },
    d: { correct: false, feedback: "Incorrect. Removing a relevant variable increases bias. It does not help." }
  };
  const selected = document.querySelector('input[name="q1"]:checked');
  const feedback = document.getElementById("quiz1-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Please select an answer.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Exercise 2: Sampling Bias
```{raw} html
<div class="quiz-container" id="quiz2">
  <p class="question">In Chapter 2 the biased model was trained only on young workers. Why did it perform so badly for older workers at test time?</p>
  <form id="quiz2-form">
    <label><input type="radio" name="q2" value="a"> The model had too many parameters and overfit to the young workers</label>
    <label><input type="radio" name="q2" value="b"> The decision tree had never seen high-experience workers during training, so it could not predict their wages</label>
    <label><input type="radio" name="q2" value="c"> Older workers' wages are simply impossible to predict from the available features</label>
    <label><input type="radio" name="q2" value="d"> The test set contained too few older workers to evaluate properly</label>
    <button type="button" onclick="checkQ2()">Submit Answer</button>
    <p id="quiz2-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ2() {
  const answers = {
    a: { correct: false, feedback: "Incorrect. Overfitting means the model fits training noise too closely. That is a separate problem from sampling bias." },
    b: { correct: true,  feedback: "Correct! A decision tree only learns rules from the data it has seen. Because no middle-aged or older worker appeared in the biased training set, the model's leaves only captured wage levels typical of young workers ($10–16/hr). Any test worker with 20+ years of experience ended up in those leaves and received a young-worker prediction far below the actual wage." },
    c: { correct: false, feedback: "Incorrect. The balanced model predicted older workers' wages accurately. The problem was the training data, not the task." },
    d: { correct: false, feedback: "Incorrect. The test set was the same for both models; the difference came entirely from the training data." }
  };
  const selected = document.querySelector('input[name="q2"]:checked');
  const feedback = document.getElementById("quiz2-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Please select an answer.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Exercise 3: Survivorship Bias
```{raw} html
<div class="quiz-container" id="quiz3">
  <p class="question">In Chapter 3, the company archive was missing short-tenure workers with below-median wages. What best describes the effect on a model trained on this archive?</p>
  <form id="quiz3-form">
    <label><input type="radio" name="q3" value="a"> The model became fairer because low-earners were excluded</label>
    <label><input type="radio" name="q3" value="b"> The model performed well on long-tenure workers but poorly on short-tenure workers, whose wage patterns it had rarely seen</label>
    <label><input type="radio" name="q3" value="c"> The model overestimated wages for long-tenure workers</label>
    <label><input type="radio" name="q3" value="d"> There was no effect because tenure was included as a feature</label>
    <button type="button" onclick="checkQ3()">Submit Answer</button>
    <p id="quiz3-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ3() {
  const answers = {
    a: { correct: false, feedback: "Incorrect. Excluding low earners does not make the model fairer. It makes it blind to an important part of the population." },
    b: { correct: true,  feedback: "Correct! The archive contained mostly workers who had stayed because they earned well. Short-tenure, low-wage workers had largely left and were invisible to the model. When the model was tested on the true population (including those who would have left), its errors for the short-tenure group were much larger than for the long-tenure group." },
    c: { correct: false, feedback: "Incorrect. The model had plenty of data on long-tenure workers, so its predictions for them were relatively accurate." },
    d: { correct: false, feedback: "Incorrect. Including tenure as a feature does not compensate for the fact that most short-tenure, low-wage workers simply did not appear in the training data." }
  };
  const selected = document.querySelector('input[name="q3"]:checked');
  const feedback = document.getElementById("quiz3-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Please select an answer.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Exercise 4: P-hacking
```{raw} html
<div class="quiz-container" id="quiz4">
  <p class="question">In Chapter 3, we tested 20 completely random yes/no features for a wage effect and found one or two "significant" results (p &lt; 0.05). What is the correct interpretation?</p>
  <form id="quiz4-form">
    <label><input type="radio" name="q4" value="a"> The significant features genuinely affect wages and should be reported</label>
    <label><input type="radio" name="q4" value="b"> The result confirms that some personal traits do influence earnings</label>
    <label><input type="radio" name="q4" value="c"> With 20 independent tests at a 5% threshold, roughly one false positive is expected by chance.</label>
    <label><input type="radio" name="q4" value="d"> The p-value threshold should have been set at 1% instead of 5%</label>
    <button type="button" onclick="checkQ4()">Submit Answer</button>
    <p id="quiz4-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ4() {
  const answers = {
    a: { correct: false, feedback: "Incorrect. The features were generated randomly and have no real relationship to wages. A significant p-value only means the result is unlikely under the null hypothesis. It does not prove a real effect, especially when many tests were run." },
    b: { correct: false, feedback: "Incorrect. The features (e.g. 'owns a pet', 'prefers mountains') were assigned randomly and cannot reflect genuine wage effects." },
    c: { correct: true,  feedback: "Correct! With 20 tests and a 5% threshold, we expect 20 × 0.05 = 1 false positive even when nothing is truly significant. Reporting only the 'significant' result, without mentioning the other 19 tests, is p-hacking. The standard remedy is to report all tests and, when running many comparisons, apply a correction such as the Bonferroni adjustment." },
    d: { correct: false, feedback: "Partially right that a stricter threshold reduces false positives, but that alone does not solve the problem of running many tests and reporting only the significant ones. The core issue is transparency about all tests performed." }
  };
  const selected = document.querySelector('input[name="q4"]:checked');
  const feedback = document.getElementById("quiz4-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Please select an answer.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

---

## Summary

Across the three preceding chapters, we encountered four distinct ways a model can
produce systematically wrong results:

1. **Omitted variable bias** arises when a variable that genuinely affects the
   outcome is left out of the model. The model cannot account for group differences
   it cannot see, and its errors become patterned rather than random.

2. **Sampling bias** arises when the training data does not reflect the population
   the model will be applied to. Groups that appear rarely in training will be
   predicted poorly and the overall accuracy score may hide this entirely.

3. **Survivorship bias** is a special case of sampling bias: only the "survivors"
   of some selection process appear in the data. The model learns about successful
   or stable cases and cannot generalise to those that did not survive.

4. **P-hacking** is not a bias in the data but in the analysis: running many tests
   and reporting only the significant ones inflates the false-positive rate, leading
   to findings that do not replicate.

The common thread is that **what is absent from the data or the report matters as
much as what is present**. Asking "who or what is missing here, and why?" is one
of the most powerful questions a researcher or practitioner can ask before trusting
a model's output.