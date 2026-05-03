# Matrix Multiplication Deep-Dive

Deck 03 of the [Linear Algebra for AI / ML](https://github.com/BrendanJamesLynskey/LLM_Hub_Linear_Algebra) series.

**Live presentation:** https://brendanjameslynskey.github.io/Linear_Algebra_AI_03_Matrix_Multiplication/

The same product, computed four different ways. Block matmul, batched matmul, the GEMM that dominates LLM compute, and the arithmetic-intensity argument that explains every choice in modern AI hardware. Includes an interactive 4&times;4 matmul visualiser that animates each of the four views.

## What's inside

- The textbook formula and what it hides
- View 1 &mdash; inner products (attention scores, logits)
- View 2 &mdash; columns as combinations of $A$'s columns (forward pass on a batch)
- View 3 &mdash; rows as combinations of $B$'s rows (attention output as weighted sum of values)
- View 4 &mdash; sum of $k$ outer products (rank bound, SVD, LoRA)
- Block matmul and the recursive structure that powers all fast implementations
- Batched matmul (BMM) and where it shows up inside multi-head attention
- Properties: associativity, distributivity, transpose, non-commutativity &mdash; with the linear-attention example
- Interactive 4&times;4 step-through that animates each of the four views
- Arithmetic intensity and the roofline argument behind tensor-core-class hardware

Single-page HTML, KaTeX-rendered maths, no build step. Open `index.html` directly.
