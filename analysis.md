# FINAL EXAM ANALYSIS
**CIIC 5015: Artificial Intelligence**
**University of Puerto Rico – Mayagüez**
**June 4, 2026**

---

## Programming Component

### Task 1: Output

```
> python models.py

Running shape test for AttentionBlock...
  Input  shape: torch.Size([2, 8, 16])
  Output shape: torch.Size([2, 8, 16])
Shape test passed!

Running causal mask verification...
  Causal mask verified: early positions unaffected by future token changes.

All tests passed!
```

```
> python gpt_model.py

Running shape tests...

  Testing Transformer_Block...
  Transformer_Block: torch.Size([4, 16, 32]) -> torch.Size([4, 16, 32])  OK

  Testing GPT...
  GPT forward: torch.Size([4, 16]) -> torch.Size([4, 16, 50])  OK

  Testing generate()...
  generate: torch.Size([1, 1]) -> torch.Size([1, 11])  OK

All shape tests passed!
```

---

### Task 2: Control Group

All default hyperparameter values (BLOCK_SIZE=64, LAYER_SIZE=64, N_LAYERS=2, LEARNING_RATE=3e-4, MAX_ITERS=3000, BATCH_SIZE=32).

#### Output

```
Loading corpus: input.txt
  Total characters : 1,115,394
  Vocabulary size  : 65 unique characters
  Characters       : "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZa"...
  Training tokens  : 1,003,854
  Validation tokens: 111,540

Model parameters : 41,921
Training for     : 3000 steps

  Step    Train Loss    Val Loss
------------------------------------
     0        4.1743      4.1739
   300        2.5734      2.5748
   600        2.4632      2.4766
   900        2.4097      2.4362
  1200        2.3889      2.4059
  1500        2.3673      2.3889
  1800        2.3583      2.3800
  2100        2.3479      2.3684
  2400        2.3321      2.3460
  2700        2.3183      2.3444
  2999        2.3158      2.3339

============================================================
GENERATED SAMPLE (500 characters):
============================================================
Fre: mud; to so Ditoor's psind s, weriouger r an?
Ro he ad I angery trelinchiveses?
My mach thefo haloul,

ICORGy ck itod whirg.
MIANG I:
Doks PO O:
Whand burwofoutee 'lad wng wet thiss be whead peer sowhe,
Non yo arof hall, he marer
Tor tharssuruchicked eve t, ime mas merd cavath wor mes woo; thed the, ar ono cend touee thas re thay a mior
Bue ntean sene shat tof, theain mo ngan.
Sount shen Sonsalare I wesit w che bryestheld bloules!, aid st E thinghist'de
Juchth illans wicher m me
I fo howen s
============================================================

Training complete.
```

---

### Task 2: Treatment Group

Changed LAYER_SIZE from 64 to 128 and N_LAYERS from 2 to 4. This increases model capacity (from 41,921 to 281,665 parameters) while keeping all other hyperparameters identical to the control group.

#### Output

```
Loading corpus: input.txt
  Total characters : 1,115,394
  Vocabulary size  : 65 unique characters
  Characters       : "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZa"...
  Training tokens  : 1,003,854
  Validation tokens: 111,540

Model parameters : 281,665
Training for     : 3000 steps  (LAYER_SIZE=128, N_LAYERS=4)

  Step    Train Loss    Val Loss
------------------------------------
     0        4.2368      4.2372
   300        2.4314      2.4420
   600        2.3330      2.3407
   900        2.2629      2.2981
  1200        2.2150      2.2569
  1500        2.1611      2.2136
  1800        2.1154      2.1709
  2100        2.0729      2.1450
  2400        2.0345      2.1282
  2700        2.0108      2.1037
  2999        1.9784      2.0685

============================================================
GENERATED SAMPLE (500 characters):
============================================================
FRGENE:
Whick thunt whid bothiour with wen'th is his ovird,
To atouser on sicet a havioish dno my fangist a---
I dithathent,
y heer tosuist, Haply have dis are dritong knowvesss,
O, yowast shrinde and of a grotw, too this so heenree;
I'le heme that had madve ponai's coungmuine. aion I astart angre at as to--mou

Beant shawa he, uploy's age betrough of hour ave;
And, ben is he was er'st boose bait
Tut four uitem hand preapentes, as ain thee irante'd!
Fout 'I ot hounds ondowod It ithils on tares
An
============================================================

Training complete.
```

---

### Task 2 Observations

All Task 1 verifications were successful.

For Task 2, the Shakespeare corpus was chosen because it is a well-known English literary style with distinct formatting patterns (character names followed by colons, verse structure, archaic vocabulary) that make it easy to qualitatively evaluate model output.

The control group used all default settings. The treatment group doubled the embedding dimension (LAYER_SIZE: 64 → 128) and doubled the number of Transformer blocks (N_LAYERS: 2 → 4), increasing total parameters by approximately 6.7× (41,921 → 281,665). All other hyperparameters remained identical.

The treatment group reached a significantly lower final loss (train: 1.9784, val: 2.0685) compared to the control group (train: 2.3158, val: 2.3339) in the same 3,000 steps. The generated text from the treatment group shows improved word boundaries and more recognizable English words (e.g., "Whick thunt", "Haply have", "And, ben is he"). This confirms that model capacity — not just training duration — has a strong effect on output quality.

---

## Concept Review Component

### Attention

**Explain the five steps of your `forward()` method in order.**

- **Step 1:** Project the input x into query (Q), key (K), and value (V) matrices using three separate learned linear layers.
- **Step 2:** Compute scaled dot-product attention scores: multiply Q by K transposed and divide by √d_k to prevent vanishing gradients in the softmax.
- **Step 3:** Apply the causal mask by filling upper-triangle positions with −∞, ensuring each position can only attend to itself and earlier positions.
- **Step 4:** Apply softmax over the last dimension to convert scores into a probability distribution (attention weights).
- **Step 5:** Compute the weighted sum of the value vectors V using the attention weights to produce the final output.

**Why do we divide by √d_k? What happens without it?**

When d_k is large, the dot products grow large in magnitude, pushing the softmax into regions where gradients become extremely small. Dividing by √d_k keeps the scores at a moderate scale so the softmax remains in a well-behaved gradient region during training.

**Why must the causal mask be applied before softmax?**

The mask replaces future positions with −∞ so that after softmax those positions become exactly 0. If the mask were applied after softmax, future positions would still receive non-zero attention weight, allowing the model to cheat by looking ahead during both training and generation.

**Why use `float('-inf')` instead of 0 for masked positions?**

A value of −∞ passed through softmax produces exactly 0, completely eliminating attention to masked positions. Using 0 would instead produce a small but nonzero weight (e^0 = 1 contributing to the softmax denominator), meaning the model would still attend to future tokens.

**What shape does the scores tensor have at each step?**

After Step 2, the scores tensor has shape `(batch_size, seq_len, seq_len)` — one score per pair of positions in the sequence.

---

### Transformer Block and GPT

**Walk through the six steps of `Transformer_Block.forward()`.**

- **Step 1:** Apply self-attention to input x → `attn_out`.
- **Step 2:** Residual connection — add `attn_out` to the original x → `x2`.
- **Step 3:** Apply the first LayerNorm to `x2` → `x3`.
- **Step 4:** Pass `x3` through the feed-forward linear layer and apply ReLU → `ff_out`.
- **Step 5:** Residual connection — add `ff_out` to `x3` (not x) → `x5`.
- **Step 6:** Apply the second LayerNorm to `x5` → final output.

**What is a residual connection and why does it matter?**

A residual connection adds a layer's input directly to its output (output = layer(x) + x). This creates a shortcut path for gradients to flow during backpropagation without passing through the transformation, which prevents the vanishing gradient problem and allows much deeper networks to train effectively.

**Why is there no activation after the final linear layer in `GPT.forward()`?**

The final linear layer produces raw logits — unnormalized scores over the vocabulary. The loss function (`F.cross_entropy`) and the `generate()` method both apply softmax internally, so adding an activation here would be redundant and would distort the logit values.

**How does `generate()` produce text autoregressively at each step?**

At each step, it crops the current sequence to the last `block_size` tokens, runs a forward pass to get logits, extracts only the logits at the final position, applies softmax to get probabilities, samples one token index from that distribution, and appends it to the sequence. This repeats `max_new_tokens` times, growing the sequence one character at a time.

**Why did you choose your corpus and what did the model learn?**

The Shakespeare corpus was chosen because it has a distinctive and well-known style — structured dialogue with speaker labels, verse formatting, and archaic vocabulary — making it easy to evaluate qualitatively whether the model is capturing real patterns. The model learned character-level structure including spacing, punctuation, capitalization of speaker names followed by colons, and partial English words. It does not yet produce grammatically correct sentences, but the output is clearly Shakespeare-influenced rather than random noise.

---

## Challenge and Analysis

The experiment compared two model sizes trained for the same number of steps (3,000). The control group used the default architecture (LAYER_SIZE=64, N_LAYERS=2, ~42k parameters) while the treatment group doubled both dimensions (LAYER_SIZE=128, N_LAYERS=4, ~282k parameters).

The treatment group achieved a final validation loss of 2.0685 versus 2.3339 for the control — a reduction of ~0.27 — demonstrating that increasing model capacity produces meaningfully better character-level predictions in the same training budget. The generated text from the larger model shows more coherent word boundaries and recognizable English structure, confirming the quantitative improvement.

One key limitation observed is that the validation loss in the treatment group (2.0685) is noticeably higher than the training loss (1.9784), indicating the model has begun to overfit. Scaling model size without also increasing regularization (e.g., dropout) or training data can reduce the model's ability to generalize to unseen text. This suggests that simply making a model bigger is not sufficient on its own — regularization and dataset size must scale alongside model capacity.
