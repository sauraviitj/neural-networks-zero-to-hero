# Neural-Networks-zero-to-hero

Working through Andrej Karpathy's Neural Networks: Zero to Hero playlist.
Each folder contains a Colab notebook with my implementations and notes.

if the notebooks show error change the url by replacing "https://github.com/" with "https://nbviewer.org/github/"
```

Video 1 (done) — micrograd
You understand backprop and autograd. The engine underneath everything.
Video 2 — makemore pt1
He builds a bigram language model — predicts the next character from the previous one. This is where you first see a real training loop on real data. Directly relevant to your LLM work — every language model including LLaMA is doing a more sophisticated version of exactly this.
Video 3 — makemore pt2 (MLP)
He builds the exact MLP you just studied — but now trains it on actual language data. You'll see weights updating, loss going down, a model actually learning. This is where the training loop clicks emotionally, not just intellectually.
Video 4 — Activations & Gradients
Why training fails. Vanishing gradients, dead neurons, bad initialisation. This is what separates engineers who debug models from engineers who guess randomly when something breaks.
Video 5 — Backprop Ninja
Manual backprop by hand. Brutal but transformative. After this, nothing in PyTorch feels like magic anymore.
Video 6 — WaveNet
Deeper architectures. Less critical for your immediate goals — can go faster here.
Video 7 — Let's build GPT
The destination. Attention mechanism, transformer, the thing inside every LLM you've worked with. After this, your EMNLP work and DuoMind make sense at a completely different level.
```
## Progress
- [✅] micrograd — autograd engine from scratch
- [✅ ] makemore pt1 — bigram language model
- [ ] makemore pt2 — MLP
- [ ] makemore pt3 — activations & gradients
- [ ] makemore pt4 — backprop ninja
- [ ] makemore pt5 — WaveNet
- [ ] nanoGPT — GPT from scratch


![image-name](https://github.com/sauraviitj/neural-networks-zero-to-hero/blob/main/1%20and%202%20Micrograd/Classes%20working.png)
