# OLA MQAR Minimal Experiment Design

## Goal

Build a standalone PyTorch experiment that compares a DeltaNet-style recurrence with the proposed orthogonal linear attention (OLA) recurrence on a small MQAR / associative recall task.

## Scope

The first version is intentionally independent from `flash-linear-attention`. It validates the recurrence behavior and task signal before migrating into an FLA-style layer.

## Task

Each sample contains key-value pairs followed by a query key. The model predicts the value associated with the queried key at the final position.

Example pattern:

```text
k1 v1 k2 v2 ... kN vN <query> kq -> vq
```

## Models

Both methods use the same wrapper:

```text
Embedding -> recurrent mixer -> final-position classifier
```

Delta baseline:

```text
S_t = S_{t-1} - beta_t k_t (k_t^T S_{t-1}) + beta_t k_t v_t^T
o_t = q_t^T S_t
```

OLA:

```text
A_t = eta_t (k_e,t k_w,t^T - k_w,t k_e,t^T)
R_t = (I + A_t)^(-1)(I - A_t)
S_t = alpha_t R_t S_{t-1} + k_w,t v_t^T
o_t = q_t^T S_t
```

## Metrics

Training and evaluation report:

- loss
- final-token value accuracy
- steps/sec
- OLA orthogonality error for a sampled transition

## Follow-Up Path

After this standalone version works, create an FLA-style `OrthogonalDeltaNet` layer with a pure PyTorch recurrent mode first, then compare against FLA `DeltaNet` with matching model settings.
