# Training without occupying the project Mac

Decision recorded September 6, 2026: the user cannot dedicate this Mac to an approximately 800-hour run. The bounded learning check is complete. Full training is not started or queued, and paid compute has not been authorized.

## Recommended next step

Use a separate NVIDIA GPU for a short, capped pilot, then choose the training budget from measured throughput and development progress. The model and data stay the same; our independently implemented model still starts from random weights. A GPU speedup must be measured, not assumed from specifications or an LLM benchmark.

The current reference trainer explicitly supports CPU and Apple MPS. Before the pilot, add and validate the CUDA device path, CUDA RNG checkpointing and timing, then use a separate CUDA-enabled environment. The existing Linux `uv sync --frozen` configuration selects CPU PyTorch and must not be presented as a GPU setup. No CUDA execution has been tested or claimed on this Mac.

First benchmark the real three-second mixtures and full enrollments, including the longest references. Compare separator microbatches while keeping the effective batch at eight. Then complete the same tiny learning/reference-switch check on the selected backend. Obtain a measured estimate including full-development inference, data preparation, checkpoint storage and instance overhead before launching a longer run.

## Current public pricing and uncertainty

[Runpod's main pricing page](https://www.runpod.io/pricing) lists RTX 4090 at **US$0.74/hour**, RTX A5000 at **US$0.27/hour**, and RTX A6000 at **US$0.53/hour**. These are page quotes checked on September 6, 2026, not reserved capacity or checkout quotes. Actual deployment availability and prices must be checked in the provider console. The 4090-specific marketing page advertises a lower starting price; use the actual chosen instance quote for any authorization.

The following is a sensitivity calculation for the 347,500-update recipe at US$0.74/hour. These speeds have **not** been measured on an external GPU.

| Hypothetical seconds per update | Training-only hours | Training-only GPU cost |
| --- | ---: | ---: |
| 0.25 | 24.1 | US$17.86 |
| 0.50 | 48.3 | US$35.72 |
| 1.00 | 96.5 | US$71.43 |

Setup, evaluation and storage are additional. A US$25 budget therefore cannot be promised to complete 100 epochs. A staged run can stop for review at complete checkpoints while preserving the original learning-rate schedule; unfinished epochs must remain reported as unfinished.

[Runpod's billing documentation](https://docs.runpod.io/pods/pricing) says compute is billed while a Pod runs and storage can continue to accrue after a Pod stops. Stopping Python alone does not stop instance billing. Before paying, the concrete deployment must include an approved total limit, provider-level stop behavior, persistent checkpoint location and retrieval plan. Do not rely on an unverified script or an idle notebook to enforce a billing cap.

## Free alternative

[Google Colab's official FAQ](https://research.google.com/colaboratory/faq.html) offers free GPU access but does not guarantee hardware or quota. Free notebooks can run for at most twelve hours, depending on availability and usage. This can support initial CUDA learning and speed checks; a complete long run may require repeated checkpoint restores and cannot have a reliable promised finish date. Colab availability limits must be respected.

Borrowing a suitable GPU machine is another possibility if the user has access to one. Its memory and measured throughput still determine feasibility. The project Mac can remain the editing, evaluation and listening machine while training runs elsewhere.
