# Homework: reading and editing neural networks through their weights

Inspired by the experiments in [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), where an LLM identified which game a policy network was trained on from its weights alone, and fixed the thinking loops of Qwen3.5-0.8B by editing 0.016% of its weights. Here you follow the same route **by hand**, with methods from this course. Using an LLM to help you is allowed, but every conclusion must be backed by a figure or a number you produced.

| Part | Notebook | Runtime |
|---|---|---|
| A. Weight detective: which game did each network play? | [HW-Part-A-Weight-Detective](HW-Part-A-Weight-Detective.ipynb) | CPU is enough |
| B. Fixing a model's thinking loops by editing its weights | [HW-Part-B-Fixing-Loops](HW-Part-B-Fixing-Loops.ipynb) | Colab T4 (about 40 minutes without your own experiments; one generation evaluation takes about 15 minutes) |

Each part is a guided investigation: run the provided steps, fill in the `TODO`s, and answer the 📌 questions in your report.

* `data/`: the policy networks for Part A
* `hwtools.py`: helper functions for Part B (loading the traces, labelling loops, rank-one edits, evaluation)

Part B uses the unedited model's generations released by the paper's authors ([RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), CC BY 4.0).
