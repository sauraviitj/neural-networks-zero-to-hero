
Batch Normalization
Batch Normalization helps neural networks train faster and more stably by normalizing activations inside the network.

It keeps each layer from seeing wildly changing input distributions during training, which makes optimization easier.

Why BatchNorm is needed
During training, the values flowing through a network keep changing as earlier weights update. This makes learning unstable and can slow training down.

BatchNorm reduces this problem by keeping activations in a controlled range before they are passed to the next layer.

The core idea
For a layer output 
z
z, BatchNorm does this for each neuron:

μ
B
=
1
m
∑
i
=
1
m
z
i
μ 
B
​
 = 
m
1
​
  
i=1
∑
m
​
 z 
i
​
 
σ
B
2
=
1
m
∑
i
=
1
m
(
z
i
−
μ
B
)
2
σ 
B
2
​
 = 
m
1
​
  
i=1
∑
m
​
 (z 
i
​
 −μ 
B
​
 ) 
2
 
z
^
i
=
z
i
−
μ
B
σ
B
2
+
ϵ
z
^
  
i
​
 = 
σ 
B
2
​
 +ϵ
​
 
z 
i
​
 −μ 
B
​
 
​
 
Then it applies learnable scale and shift:

y
i
=
γ
z
^
i
+
β
y 
i
​
 =γ 
z
^
  
i
​
 +β
μ
B
μ 
B
​
 : mini-batch mean.

σ
B
2
σ 
B
2
​
 : mini-batch variance.

ϵ
ϵ: small constant for numerical stability.

γ
γ: learnable scale.

β
β: learnable shift.

Layer-wise or neuron-wise?
BatchNorm is applied per neuron / per feature, but computed across the batch.

So if your activation matrix is shape (batch_size, n_hidden), BatchNorm computes one mean and one std for each hidden neuron using all examples in that batch.

Picture
text
Batch shape: (32, 200)

For neuron 1:  use all 32 values → mean, std → normalize
For neuron 2:  use all 32 values → mean, std → normalize
...
For neuron 200: use all 32 values → mean, std → normalize
Why gamma and beta exist
If we only normalized to zero mean and unit variance, the network would become too constrained.
So we add:

γ
γ: lets the network stretch or shrink the normalized values.

β
β: lets the network shift them left or right.

This means BatchNorm normalizes first, then gives the model freedom to recover the best scale and offset.

Training vs testing
During training, BatchNorm uses the mean and std of the current mini-batch.

During testing, batch statistics are not reliable, especially if the batch is small or size 1. So BatchNorm uses running averages of mean and std collected during training.

Exponential moving average
r
u
n
n
i
n
g
_
m
e
a
n
=
m
o
m
e
n
t
u
m
⋅
r
u
n
n
i
n
g
_
m
e
a
n
+
(
1
−
m
o
m
e
n
t
u
m
)
⋅
b
a
t
c
h
_
m
e
a
n
running_mean=momentum⋅running_mean+(1−momentum)⋅batch_mean
r
u
n
n
i
n
g
_
s
t
d
=
m
o
m
e
n
t
u
m
⋅
r
u
n
n
i
n
g
_
s
t
d
+
(
1
−
m
o
m
e
n
t
u
m
)
⋅
b
a
t
c
h
_
s
t
d
running_std=momentum⋅running_std+(1−momentum)⋅batch_std
These running values are then used at inference time.

Small example
Suppose one neuron outputs these values for a batch of 4 samples:

text
z = [2, 4, 6, 8]
Step 1: mean
μ
B
=
2
+
4
+
6
+
8
4
=
5
μ 
B
​
 = 
4
2+4+6+8
​
 =5
Step 2: variance
σ
B
2
=
(
2
−
5
)
2
+
(
4
−
5
)
2
+
(
6
−
5
)
2
+
(
8
−
5
)
2
4
=
5
σ 
B
2
​
 = 
4
(2−5) 
2
 +(4−5) 
2
 +(6−5) 
2
 +(8−5) 
2
 
​
 =5
Step 3: normalize
z
^
=
z
−
5
5
+
ϵ
z
^
 = 
5+ϵ
​
 
z−5
​
 
So the values become centered around 0 and scaled to a healthier range.

Step 4: scale and shift
y
=
γ
z
^
+
β
y=γ 
z
^
 +β
If 
γ
=
1
γ=1 and 
β
=
0
β=0, you keep the normalized values.
If the network learns better values, it can stretch or move them as needed.

Simple flow
text
Input -> Linear layer -> BatchNorm -> tanh/ReLU -> next layer
During training
text
use batch mean/std
update running mean/std
learn gamma and beta
During testing
text
use running mean/std
apply gamma and beta
One-line summary
BatchNorm normalizes each neuron across the mini-batch, then uses 
γ
γ and 
β
β to re-scale it, and uses running averages of mean/std at test time
