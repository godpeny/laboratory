# Why is gradient with respect to bias equal to sum of incoming gradients?

It is not difficult to derive gradients `dx` and `dw` and implement them in Python. The gradient `db`, however, requires careful thought. The expression `db = np.sum(dout, axis=0)` is not easy to understand.

First, you need to understand what really happens inside this simple equation $$out = xw + b$$. One way to understand this is by analyzing dimensions of the matrices. That is, $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(1 \times M)}b$$.

Do you notice incompatibility of $$b$'s dimension and $$out$$'s? Think of it for a second and read on!

What happens in the expression is broadcast of $$b$$ from $$1 \times M$$ to $$N \times M$$. Intuitively, there is only one set of bias of size $$M$$ which shares across all examples $$x$$. In fact, there is an *invisible input* for the bias term. That is, $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(N \times 1)}{ix}\underset{(1 \times M)}b$$ where $$ix$$ contains all 1's and stays there just for the sake of facilitating later calculation. (As the term suggests, bias does not depend on the input.)

That's it for `layer_forward`. Now, let's proceed to `layer_backward`.

Considering $$\underset{(N \times M)}{out} = \underset{(N \times D)}x\underset{(D \times M)}w + \underset{(N \times 1)}{ix}\underset{(1 \times M)}b$$, we can derive $$db$$ (in short, for $$\frac{dLoss}{db}$$) using a chain rule with $$\underset{(1 \times M)}{db} = \underset{(1 \times N)}{ix^\mathsf{T}} \underset{(N \times M)}{dout}$$. Recall that $$ix$$ is all 1's; therefore, the expression just sums $$dout$$ along axis = 0, that is, `db = np.sum(dout, axis=0)`.

Not really trivial, is it?

## Reference
https://www.reddit.com/r/cs231n/comments/em7d8l/why_is_gradient_with_respect_to_b_equal_to_sum_of/
https://stackoverflow.com/questions/72924521/why-we-used-the-sum-in-the-code-for-the-gradient-of-the-bias-and-why-we-didnt-i
https://github.com/nimitpattanasri/mlxai.github.io/blob/master/_posts/2017-01-10-a-modular-approach-to-implementing-fully-connected-neural-networks.md?plain=1


