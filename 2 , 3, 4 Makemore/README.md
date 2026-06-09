# 📐 Batch Normalization — Complete Notes

Batch Normalization helps neural networks train faster and more stably
by keeping activations in a healthier range at each layer during training.

---

## 🧠 Why BatchNorm is needed

During training, weights keep changing, so the input distribution to each
layer keeps shifting too. This is called **internal covariate shift** and
it makes learning slow and unstable.

BatchNorm fixes this by normalizing activations at each layer so the next
layer always sees inputs with a stable distribution.

---

## ⚙️ The core formula

For a layer output `z`, BatchNorm does this **per neuron**, across the batch:

```text
Step 1 — Batch mean:
  mu  = (1/m) * sum(z_i)

Step 2 — Batch variance:
  var = (1/m) * sum((z_i - mu)^2)

Step 3 — Normalize:
  z_norm = (z - mu) / sqrt(var + epsilon)

Step 4 — Scale and shift:
  y = gamma * z_norm + beta
```

| Symbol    | Meaning                                  |
|-----------|------------------------------------------|
| `mu`      | mean of the mini-batch for that neuron   |
| `var`     | variance of the mini-batch               |
| `epsilon` | small number to avoid dividing by zero   |
| `gamma`   | learnable scale (trained by backprop)    |
| `beta`    | learnable shift (trained by backprop)    |

---

## 🔢 Per neuron, across the batch

BatchNorm is applied per neuron, computed across the batch.

If activation shape is `(batch_size, n_hidden)`:

```text
Batch shape: (32, 200)

neuron 1   → look at all 32 values → compute mean, std → normalize
neuron 2   → look at all 32 values → compute mean, std → normalize
...
neuron 200 → look at all 32 values → compute mean, std → normalize
```

Each neuron gets its own mean and std — not one global mean for all neurons.

---

## 🎛️ Why gamma and beta exist

If we only normalized to zero mean and unit variance, the network would
be too constrained to represent useful patterns.

- `gamma` → lets the network stretch or shrink the values
- `beta`  → lets the network shift values left or right

This means BatchNorm normalizes first, then lets the model recover
the best scale and offset by learning `gamma` and `beta` via gradient descent.

---

## 🔁 Simple flow

```text
Input
  |
  v
Linear Layer  (z = X @ W + b)
  |
  v
BatchNorm     (normalize z, then apply gamma and beta)
  |
  v
Activation    (tanh / ReLU)
  |
  v
Next Layer
```

---

## 📊 Small example (batch size = 4)

Suppose one neuron outputs:

```text
z = [2, 4, 6, 8]
```

**Step 1 — mean:**

```text
mu = (2 + 4 + 6 + 8) / 4 = 5
```

**Step 2 — variance:**

```text
var = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4
    = (9 + 1 + 1 + 9) / 4
    = 5
```

**Step 3 — normalize:**

```text
z_norm = (z - 5) / sqrt(5 + eps)
       ≈ [-1.34, -0.45, +0.45, +1.34]
```

**Step 4 — scale and shift:**

```text
y = gamma * z_norm + beta
```

If gamma=1 and beta=0  →  values stay as z_norm
As training progresses →  gamma and beta adjust to best values

---

## 🕒 Training vs Testing

| | Training | Testing |
|---|---|---|
| Mean used   | current batch mean  | running mean (EMA) |
| Std used    | current batch std   | running std (EMA)  |
| gamma/beta  | updated by backprop | fixed              |

---

## 📈 Exponential Moving Average (EMA) for Test Time

During testing you may have only one example — no batch to compute stats from.
So during training, BatchNorm keeps a **running estimate** using EMA:

```text
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_std  = momentum * running_std  + (1 - momentum) * batch_std
```

> In Karpathy's notebook: momentum = 0.999

**What momentum means:**
- New batch stats contribute only 0.1% each step
- The running estimate slowly tracks the true average across the whole dataset
- Old estimates are forgotten very slowly (0.999 weight to old value)

**Visual:**

```text
Batch 1  → batch_mean_1 ─┐
Batch 2  → batch_mean_2 ─┼─► EMA → running_mean (used at test time)
Batch 3  → batch_mean_3 ─┘
...
```

**At inference time:**

```text
z_norm = (z - running_mean) / (running_std + epsilon)
y      = gamma * z_norm + beta
```

No batch needed. Uses stable running estimates built during training.

---


## ✅ One-line summary

> BatchNorm normalizes each neuron across the mini-batch, applies learnable
> `gamma` and `beta` to let the network recover the best scale, and uses
> exponential moving averages of mean/std at test time for stable inference.

![image-name](https://github.com/sauraviitj/neural-networks-zero-to-hero/blob/main/2%20%2C%203%2C%204%20Makemore/batchnorm.png)