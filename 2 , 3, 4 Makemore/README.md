# Batch Normalization

Batch Normalization helps neural networks train faster and more stably by keeping activations in a healthier range during training.
It reduces the problem of changing activation distributions inside the network, which makes optimization easier.

---

## Why BatchNorm is needed

During training, the values flowing through a network keep changing as earlier weights update.
This makes learning unstable and can slow training down.
BatchNorm helps by keeping activations well-behaved before they are passed to the next layer.

---

## The core idea

For a layer output \(z\), BatchNorm does this for each neuron:

\[
\mu_B = \frac{1}{m}\sum_{i=1}^{m} z_i
\]

\[
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(z_i - \mu_B)^2
\]

\[
\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\]

Then it applies learnable scale and shift:

\[
y_i = \gamma \hat{z}_i + \beta
\]

- \(\mu_B\): mini-batch mean.
- \(\sigma_B^2\): mini-batch variance.
- \(\epsilon\): small constant for numerical stability.
- \(\gamma\): learnable scale.
- \(\beta\): learnable shift.

---

## Layer-wise or neuron-wise?

BatchNorm is applied per neuron / per feature, but computed across the batch.
So if your activation matrix is shape `(batch_size, n_hidden)`, BatchNorm computes one mean and one std for each hidden neuron using all examples in that batch.

### Picture

```text
Batch shape: (32, 200)

For neuron 1:  use all 32 values → mean, std → normalize
For neuron 2:  use all 32 values → mean, std → normalize
...
For neuron 200: use all 32 values → mean, std → normalize
```

---

## Why gamma and beta exist

If we only normalized to zero mean and unit variance, the network would become too constrained.
So we add:

- \(\gamma\): lets the network stretch or shrink the normalized values.
- \(\beta\): lets the network shift them left or right.

This means BatchNorm normalizes first, then gives the model freedom to recover the best scale and offset.

---

## Training vs testing

During training, BatchNorm uses the mean and std of the current mini-batch.
During testing, batch statistics are not reliable, especially if the batch is small or size 1.
So BatchNorm uses running averages of mean and std collected during training.

---

## Exponential moving average

BatchNorm stores the statistics with an exponential moving average:

```text
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_std  = momentum * running_std  + (1 - momentum) * batch_std
```

- `momentum` close to 1 means slow, stable updates.
- `momentum` close to 0 means faster updates, but noisier estimates.

### At inference time

BatchNorm uses:

```text
z_norm = (z - running_mean) / (running_std + epsilon)
y = gamma * z_norm + beta
```

So:
- batch mean/std → used during training.
- running mean/std → used during testing.

---

## Small example

Suppose one neuron outputs these values for a batch of 4 samples:

```text
z = [2, 4, 6, 8]
```

### Step 1: mean

\[
\mu_B = \frac{2+4+6+8}{4} = 5
\]

### Step 2: variance

\[
\sigma_B^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = 5
\]

### Step 3: normalize

\[
\hat{z} = \frac{z - 5}{\sqrt{5 + \epsilon}}
\]

So the values become centered around 0 and scaled to a healthier range.

### Step 4: scale and shift

\[
y = \gamma \hat{z} + \beta
\]

If \(\gamma = 1\) and \(\beta = 0\), you keep the normalized values.
If the network learns better values, it can stretch or move them as needed.

---

## Simple flow

```text
Input -> Linear layer -> BatchNorm -> tanh/ReLU -> next layer
```

### During training

```text
use batch mean/std
update running mean/std
learn gamma and beta
```

### During testing

```text
use running mean/std
apply gamma and beta
```

---

## One-line summary

BatchNorm normalizes each neuron across the mini-batch, then uses \(\gamma\) and \(\beta\) to re-scale it, and uses running averages of mean/std at test time.
