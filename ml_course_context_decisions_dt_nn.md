# Machine Learning Course Context: Decisions, Decision Trees, and Neural Networks

This context file summarizes only the theoretical and practical scope contained in the three provided course lectures:

- `MLclass5Decision.pdf`: Bayesian Decision Theory / Decisions
- `MLclass6DT.pdf`: Decision Trees
- `MLclass7NN.pdf`: Neural Networks

Use this file as a curriculum-aligned reference for an AI coding assistant. Do not assume techniques, terminology, formulas, or implementation requirements that are not represented here.

---

## 1. Bayesian Decision Theory / Decisions

### 1.1 Purpose and framing

Bayesian Decision Theory is presented as a fundamental approach in pattern classification. Its purpose is to quantify classification decisions using probabilities. Decisions may be based only on prior probabilities when those are the only available information, or, more commonly, on class-conditional probability density functions combined with prior probabilities through Bayes' theorem.

Core notation:

- Pattern / feature vector: `x`, or multidimensional `x ∈ R^d`.
- States of nature / classes: `ω_i`, where `i = 1,...,c`.
- Class prior: `P(ω_i)`.
- Class-conditional density: `p(x | ω_i)`.
- Posterior probability: `P(ω_i | x)`.
- Evidence / marginal density: `p(x)`.

Bayes' theorem in the multidimensional, multiclass setting:

```text
P(ω_j | x) = p(x | ω_j) P(ω_j) / p(x)

p(x) = Σ_{j=1}^{c} p(x | ω_j) P(ω_j)
```

When only the classification decision is needed, `p(x)` can be ignored because it is common to all classes for a fixed `x`.

---

### 1.2 Two-class minimum-error decision rule

For a two-class problem, after observing a pattern `x`, the probability of making an erroneous decision is:

```text
P(error | x) = P(ω_1 | x), if deciding ω_2
P(error | x) = P(ω_2 | x), if deciding ω_1
```

To minimize this error for a specific `x`, use the posterior comparison rule:

```text
Decide ω_1 if P(ω_1 | x) > P(ω_2 | x)
Otherwise decide ω_2
```

The average probability of error is:

```text
P(error) = ∫ P(error, x) dx
         = ∫ P(error | x) p(x) dx
```

If `P(error | x)` is minimized for every `x`, then the overall `P(error)` is minimized.

For the two-class case:

```text
P(error | x) = min[P(ω_1 | x), P(ω_2 | x)]
```

Using priors and class-conditionals, the decision rule can be written as:

```text
Decide ω_1 if p(x | ω_1) P(ω_1) > p(x | ω_2) P(ω_2)
Otherwise decide ω_2
```

Special cases mentioned:

- If `p(x | ω_1) = p(x | ω_2)`, the decision depends on the priors.
- If `P(ω_1) = P(ω_2)`, the decision depends on the class-conditional densities.

---

### 1.3 Bayes optimal error in two-class 1D classification

Let `R_1` be the decision region assigned to `ω_1`, and `R_2` the decision region assigned to `ω_2`. The total error is:

```text
P(error) = P(x ∈ R_2, ω_1) + P(x ∈ R_1, ω_2)
         = P(x ∈ R_2 | ω_1)P(ω_1) + P(x ∈ R_1 | ω_2)P(ω_2)
         = ∫_{R_2} p(x | ω_1)P(ω_1) dx + ∫_{R_1} p(x | ω_2)P(ω_2) dx
```

The Bayes decision boundary is located where:

```text
P(ω_1 | x) = P(ω_2 | x)
```

The Bayes error rate is the optimal irreducible error under the given probability model.

---

### 1.4 Generalization beyond two-class minimum-error decisions

The lecture generalizes the discussion in four directions:

1. More than one feature: `x → x ∈ R^d`.
2. More than two classes: `ω_i`, `i = 1,...,c`.
3. Actions other than deciding the class, such as rejection / deciding not to decide.
4. A general loss / risk function instead of only probability of error.

---

### 1.5 Loss, conditional risk, and Bayes risk

Actions are denoted by `α_i`. Taking action `α_i` when the true state is `ω_j` incurs loss:

```text
λ(α_i | ω_j)
```

The conditional risk of action `α_i` given pattern `x` is the expected loss:

```text
R(α_i | x) = Σ_{j=1}^{c} λ(α_i | ω_j) P(ω_j | x)
```

A general decision rule `α(x)` chooses an action for every `x`. The overall risk is:

```text
R(α) = ∫ R(α(x) | x) p(x) dx
```

To minimize the overall risk, choose for every `x` the action with minimum conditional risk. The resulting minimum overall risk is called the Bayes risk and is presented as the best achievable performance.

---

### 1.6 Two-class risk-sensitive decision rule

Let:

```text
λ_ij = λ(α_i | ω_j)
```

where `α_i` means deciding class `ω_i`, while the true state is `ω_j`.

The conditional risks are:

```text
R(α_1 | x) = λ_11 P(ω_1 | x) + λ_12 P(ω_2 | x)
R(α_2 | x) = λ_21 P(ω_1 | x) + λ_22 P(ω_2 | x)
```

Decide `ω_1` if:

```text
R(α_1 | x) < R(α_2 | x)
```

Equivalent posterior-weighted loss comparison:

```text
(λ_21 - λ_11) P(ω_1 | x) > (λ_12 - λ_22) P(ω_2 | x)
```

Using Bayes' theorem:

```text
(λ_21 - λ_11) p(x | ω_1)P(ω_1) > (λ_12 - λ_22) p(x | ω_2)P(ω_2)
```

Assuming `λ_21 > λ_11`, the likelihood-ratio form is:

```text
p(x | ω_1) / p(x | ω_2) > [(λ_12 - λ_22) / (λ_21 - λ_11)] [P(ω_2) / P(ω_1)]
```

Thus, Bayes' decision rule recommends deciding `ω_1` if the likelihood ratio exceeds a threshold independent of `x`.

---

### 1.7 Zero-one loss and minimum-error-rate classification

In standard classification, the state of nature coincides with one of the classes, and the action `α_i` means deciding that the true state is `ω_i`.

Zero-one loss:

```text
λ(α_i | ω_j) = 0, if i = j
λ(α_i | ω_j) = 1, if i ≠ j
```

Under zero-one loss:

```text
R(α_i | x) = Σ_{j≠i} P(ω_j | x)
           = 1 - P(ω_i | x)
```

Therefore, minimizing risk is equivalent to maximizing the posterior:

```text
Decide ω_i if P(ω_i | x) > P(ω_j | x) for all j ≠ i
```

---

### 1.8 Discriminant functions

A classifier can be represented by discriminant functions:

```text
g_i(x), i = 1,...,c
```

Classification rule:

```text
Assign x to ω_i if g_i(x) > g_j(x) for all j ≠ i
```

Possible discriminant functions from the lecture:

```text
g_i(x) = -R(α_i | x)                 [general risk case]
g_i(x) = P(ω_i | x)                  [minimum-error case]
g_i(x) = p(x | ω_i)P(ω_i)
g_i(x) = ln p(x | ω_i) + ln P(ω_i)
```

For a two-class dichotomizer:

```text
Decide ω_1 if g(x) = g_1(x) - g_2(x) > 0
```

Minimum-error forms:

```text
g(x) = P(ω_1 | x) - P(ω_2 | x)

g(x) = ln[p(x | ω_1) / p(x | ω_2)] + ln[P(ω_1) / P(ω_2)]
```

---

### 1.9 Normal density discriminant functions

For minimum-error-rate classification:

```text
g_i(x) = ln p(x | ω_i) + ln P(ω_i)
```

If densities are multivariate normal:

```text
p(x) = 1 / ((2π)^{d/2} |Σ|^{1/2}) * exp[-1/2 (x - μ)^T Σ^{-1}(x - μ)]
```

Then the discriminant function is:

```text
g_i(x) = -1/2 (x - μ_i)^T Σ_i^{-1}(x - μ_i)
         - d/2 ln(2π)
         - 1/2 ln|Σ_i|
         + ln P(ω_i)
```

For normal densities, the loci of constant density are hyperellipsoids where the quadratic form is constant:

```text
r^2 = (x - μ)^T Σ^{-1}(x - μ)
```

This is the Mahalanobis distance. Principal axes of the hyperellipsoids are given by the eigenvectors of `Σ`, and their lengths are given by the eigenvalues of `Σ`.

A discriminant function for two classes having normal densities is hyperquadratic: it can form hyperplanes, hyperspheres, hyperellipsoids, or paraboloids.

---

### 1.10 Spherical equal covariance case: `Σ_i = σ²I`

This is the simplest normal-density case in the lecture. Features are statistically independent and each feature has the same variance. Clusters are hyperspherical and concentrated around the class means.

Omitting constants that are the same for all classes:

```text
g_i(x) = - ||x - μ_i||² / (2σ²) + ln P(ω_i)
```

where:

```text
||x - μ_i||² = (x - μ_i)^T (x - μ_i)
```

Interpretation:

- If `x` is equally near two different mean vectors, the optimal decision tends toward the class with the higher prior probability.
- Expanding the quadratic form gives a linear discriminant because `x^T x` is the same for all classes.

Expanded form:

```text
g_i(x) = -1/(2σ²) [x^T x - 2μ_i^T x + μ_i^T μ_i] + ln P(ω_i)
```

Linear discriminant form:

```text
g_i(x) = w_i^T x + w_i0
```

where:

```text
w_i = μ_i / σ²
w_i0 = -1/(2σ²) μ_i^T μ_i + ln P(ω_i)
```

The resulting classifier is a linear machine with decision boundaries composed of pieces of hyperplanes defined by:

```text
g_i(x) = g_j(x)
```

For two classes, the boundary can be written:

```text
w^T(x - x_0) = 0
```

where:

```text
w = μ_i - μ_j

x_0 = 1/2(μ_i + μ_j) - [σ² / ||μ_i - μ_j||²] ln[P(ω_i) / P(ω_j)] (μ_i - μ_j)
```

Geometric interpretation:

- The hyperplane separating `R_i` and `R_j` is orthogonal to the line connecting the means.
- If priors are equal, the second term vanishes and `x_0` is halfway between the means. The boundary is the perpendicular bisector between the means.
- If priors are not equal, the boundary shifts away from the more likely mean.
- If the variance is small relative to the squared distance between the means, the decision-boundary location is relatively insensitive to the priors.
- If covariance matrices are not necessarily equal, decision boundaries are not necessarily linear.

---

## 2. Decision Trees

### 2.1 Definition and representation

Decision trees are presented as widely used and practical methods for inference. They can be represented as sets of `if-then` rules, improving human readability. They are usually used to approximate discrete-valued functions, especially classification functions.

Decision tree classification process:

1. Start at the root node.
2. Each internal node specifies a test on an attribute / feature.
3. Each branch corresponds to one possible value of that attribute.
4. Move down the branch matching the instance's attribute value.
5. Repeat recursively on the subtree.
6. Stop at a leaf, which gives the class.

A root-to-leaf path represents a conjunction of attribute tests. The full tree represents a disjunction of such conjunctions.

For the PlayTennis example, the `Yes` class corresponds to:

```text
(Outlook = Sunny ∧ Humidity = Normal)
∨ (Outlook = Overcast)
∨ (Outlook = Rain ∧ Wind = Weak)
```

---

### 2.2 Suitable problems for decision trees

Decision trees are suitable for problems with the following characteristics:

- Attribute-value instance representation.
- Discrete-valued target function.
- Disjunctive descriptions may be required.
- Missing data and errors are allowed.
- In practice, this means classification problems in the lecture scope.

---

### 2.3 ID3 algorithm

ID3 stands for Iterative Dichotomiser 3. It is presented as a top-down decision-tree learning algorithm.

Main procedure:

1. Start by asking which attribute should be tested at the root.
2. Evaluate each attribute using a statistical test that measures how well it alone classifies the training set.
3. Select the best attribute for the root.
4. Create one descendant branch for each possible value of that attribute.
5. Sort training examples into descendants according to attribute values.
6. Repeat recursively for each descendant node, choosing the best remaining attribute at that node.
7. The search is greedy: it performs no backtracking.

ID3 base cases from the lecture pseudocode:

- Create a root node.
- If all examples are positive, return a single-node tree labeled `+`.
- If all examples are negative, return a single-node tree labeled `-`.
- If the attributes set is empty, return a single-node tree labeled with the most common target value among examples.
- Otherwise, choose the attribute from the available attributes that best classifies the examples.
- For each possible value `v_i` of the chosen attribute:
  - Add a branch corresponding to `A = v_i`.
  - Let `Examples_vi` be the subset of examples having `A = v_i`.
  - If `Examples_vi` is empty, attach a leaf labeled with the most common target value in the parent examples.
  - Otherwise, recursively apply ID3 to `Examples_vi`, with the chosen attribute removed from the attribute set.

---

### 2.4 Attribute selection using entropy and information gain

The best attribute is selected using information gain, a measure of expected entropy reduction.

Entropy characterizes impurity, disorder, or uncertainty of a set of examples.

For binary classification with proportions `p_+` and `p_-`:

```text
I(S) = -p_+ log₂(p_+) - p_- log₂(p_-)
```

Example from PlayTennis:

```text
Entropy([9+, 5-]) = -(9/14)log₂(9/14) - (5/14)log₂(5/14) = 0.94
```

Boundary cases:

- Entropy is `0` if all examples belong to one class.
- Define `0 log₂ 0 = 0`.
- Entropy is `1` in binary classification when `p_+ = p_-`.

For a multiclass problem with `c` classes:

```text
I(S) = -Σ_{i=1}^{c} p_i log₂(p_i)
```

where `p_i` is the proportion of examples in class `i`. The maximum entropy is:

```text
log₂(c)
```

For a 3-class variable, the lecture notes that the simplex corners represent a single outcome with entropy `0`, while the center represents a uniform distribution with entropy `1.585`.

---

### 2.5 Information gain formula

Information gain is the expected reduction in entropy caused by partitioning examples according to an attribute.

For example set `S` and attribute `A`:

```text
Gain(S, A) = I(S) - E(A)
           = I(S) - Σ_{v ∈ Values(A)} (|S_v| / |S|) I(S_v)
```

where:

- `Values(A)` is the set of possible values of attribute `A`.
- `S_v` is the subset of `S` where attribute `A` has value `v`.
- `E(A)` is the weighted entropy after splitting by `A`.

Interpretation:

```text
Gain(S, A) = expected reduction in entropy caused by knowing the value of A
```

Example values from PlayTennis:

```text
Gain(S, Wind)        = 0.048
Gain(S, Humidity)    = 0.151
Gain(S, Outlook)     = 0.246
Gain(S, Temperature) = 0.029
```

Since `Outlook` has the highest information gain, it is selected as the root. Each value of `Outlook` creates a branch, and the tree continues growing recursively.

---

### 2.6 Approximate inductive bias of ID3

ID3 approximately prefers:

- Shorter trees over longer trees.
- Trees that place high-information-gain attributes close to the root.

This bias follows from the greedy top-down choice of high-gain attributes without backtracking.

---

### 2.7 Overfitting in decision trees

ID3 grows each branch deep enough to perfectly classify the training set. This can cause overfitting when the training set is too small to represent the true target function or when the data contains noise.

Definition from the lecture:

Given hypothesis space `H`, a hypothesis `h ∈ H` overfits the training data if there exists another hypothesis `h' ∈ H` such that:

```text
h has smaller error than h' on the training set,
but h' has smaller error than h on the entire distribution of instances.
```

Reasons for overfitting:

1. Small data size and few representative instances can create unnecessary splits.
2. Noise or mislabeled examples can create unnecessary splits.

---

### 2.8 Avoiding overfitting and pruning

Two main approaches:

1. Stop growing the tree before it perfectly classifies the training set.
2. Allow overfitting, then post-prune the tree.

Two ways to determine correct tree size:

1. Use a validation set to decide which nodes to post-prune.
2. Use a criterion based on the complexity of encoding the training set and the decision tree, stopping growth when this criterion is minimized.

Validation-set pruning procedure:

1. For each decision node, remove the subtree rooted at that node.
2. Replace it with a leaf node.
3. Label the new leaf with the most common class among training examples affiliated with that node.
4. Remove a node only if the pruned tree performs no worse than the original tree on the validation set.
5. Prune iteratively, each time choosing the node whose removal most increases validation-set accuracy.
6. Stop pruning when validation-set accuracy decreases.

---

### 2.9 C4.5 compared with ID3

C4.5 extends ID3 in the lecture by:

- Handling both continuous and discrete attributes.
- Splitting continuous attributes according to thresholds.
- Handling missing attribute values by not using missing values in gain and entropy calculations.
- Handling attributes with differing costs.
- Pruning trees after creation by replacing branches that do not help with leaf nodes.

ID3 is described as using categorical predictors, while C4.5 can handle continuous attributes through threshold-based splits.

---

### 2.10 Additional decision-tree scope notes

The lecture also mentions:

- Classification by C4.5 examples.
- Decision trees as classifiers.
- Decision trees for regression.
- Decision trees can become complex.

No detailed regression-tree algorithm or formula is provided in the lecture text, so a coding assistant should not assume regression-tree implementation details beyond the fact that decision trees can be used for regression.

---

## 3. Neural Networks

### 3.1 Motivation and core properties

Neural networks are presented as powerful computational mechanisms due to:

- Simple processing units / neurons that are easy to model, compute, and implement.
- Large, though not always large, numbers of highly interconnected neurons, providing distributed computational capability for complex problems.
- Nonlinearity of hidden units, enabling solution of nonlinearly separable problems.
- Logistic sigmoid or other normalized activation functions, enabling interpretation of outputs as posterior probabilities.
- Relatively easy message propagation and learning algorithms.

The lecture gives biological motivation by relating biological neurons to artificial neurons.

---

### 3.2 Neural networks as discriminant functions

Neural networks are connected to discriminant functions. A discriminant function can be an alternative to Bayes' theorem.

For class `C_k`:

```text
y_k(x) = P(C_k | x) = p(x | C_k)P(C_k)
```

Classification rule:

```text
Assign x to C_k if y_k(x) > y_j(x) for all j ≠ k
```

Purpose:

```text
Minimize the probability of misclassification.
```

Important interpretation:

- If the discriminant function is identified with the posterior probability, the optimal decision is maximum a posteriori.
- The discriminant function is parametrized; its parameters are found by training using data.
- The simplest discriminant function is linear.
- Neural networks extend this by using linear combinations of transformed inputs.

---

### 3.3 Linear discriminant functions: two-class case

The simplest discriminant function is a linear combination of the input variables:

```text
y(x) = w^T x + w_0
```

where:

- `w` is the weight vector.
- `w_0` is the bias / negative threshold term.

The lecture states that a linear model is optimal for normal class-conditionals with equal and spherical covariance matrices.

The decision boundary is:

```text
y(x) = 0
```

In a `d`-dimensional input space, this boundary is a `(d-1)`-dimensional hyperplane. In 2D, it is a straight line.

---

### 3.4 Geometrical interpretation of linear discriminants

If `x_1` and `x_2` lie on the hyperplane:

```text
y(x_1) = y(x_2) = 0
```

Then:

```text
w^T(x_1 - x_2) = 0
```

Therefore:

- `w` is normal to any vector lying in the hyperplane.
- `w` determines the orientation of the decision boundary.

For a point `x` on the boundary, the normal distance from the origin to the hyperplane is:

```text
l = w^T x / ||w|| = -w_0 / ||w||
```

A two-class neural network with one output can represent this linear discriminant:

```text
y(x) = w^T x + w_0
```

---

### 3.5 Multiclass linear discriminants and multi-output networks

For multiclass classification, use one discriminant function per class:

```text
y_k(x) = W_k^T x + w_k0
```

Classification rule:

```text
Assign x to C_k if y_k(x) > y_j(x) for all j ≠ k
```

The decision boundary between `C_k` and `C_j` is:

```text
y_k(x) = y_j(x)
```

For linear discriminants:

```text
(W_k - W_j)^T x + (w_k0 - w_j0) = 0
```

A multi-output neural network can represent multiple linear discriminant functions, where each output is:

```text
y_k(x) = Σ_{i=1}^{d} w_ki x_i + w_k0
       = Σ_{i=0}^{d} w_ki x_i
```

The lecture describes the resulting model as a piecewise linear discriminator created by the intersection of `c` linear models.

---

### 3.6 The perceptron

The perceptron is presented as a single-layer neural network.

Perceptron output:

```text
y = g(Σ_{j=0}^{M} w_j φ_j(x)) = g(w^T φ)
```

where:

- `φ = [φ_0, ..., φ_M]` is a vector of activations.
- `g(a)` is a threshold activation:

```text
g(a) = -1, when a < 0
g(a) = +1, when a ≥ 0
```

---

### 3.7 Perceptron criterion

The lecture notes that defining error directly by the number of misclassifications is difficult because it produces a piecewise constant error function. Smooth changes in weights can move decision boundaries across data points, causing discontinuities in the error and making gradient-descent mechanisms difficult.

For class `C_1`, with target `t_n = +1`:

```text
w^T φ_n > 0
```

For class `C_2`, with target `t_n = -1`:

```text
w^T φ_n < 0
```

Equivalently, for all training vectors:

```text
w^T(φ_n t_n) > 0
```

Perceptron criterion:

```text
E_perc(w) = -Σ_{φ_n ∈ H} w^T(φ_n t_n)
```

where `H` is the set of misclassified activation vectors under the current `w`.

Properties:

- The term inside the sum is negative for misclassified patterns.
- The total expression is positive and becomes `0` when there are no errors.
- As the boundary moves, the set of misclassified patterns changes from iteration to iteration.

---

### 3.8 Perceptron learning algorithm

Algorithmic behavior:

1. Cycle through all training patterns.
2. Check whether each pattern is classified correctly under the current weights.
3. If the pattern is correctly classified, do nothing.
4. If the pattern is misclassified:
   - Add the pattern vector multiplied by the learning rate to the weight vector if the pattern is labeled class `C_1`.
   - Subtract the pattern vector multiplied by the learning rate from the weight vector if the pattern is labeled class `C_2`.

The lecture states that for linearly separable problems, the single-layer perceptron is guaranteed to converge in a finite number of steps.

---

### 3.9 Linear separability and XOR

The exclusive-OR problem is presented as nonlinearly separable.

To solve nonlinearly separable problems, the lecture motivates the need for nonlinear, adaptive activation functions, preferably differentiable.

---

### 3.10 Logistic sigmoid activation function

Neuron output with logistic sigmoid:

```text
y = g(w^T x + w_0)
```

where:

```text
g(a) = 1 / (1 + exp(-a))
     = 1 / (1 + exp(-λ(w^T x + w_0)))
```

Advantages stated in the lecture:

1. The function is bounded between `0` and `1`, so neuron output can represent probability.
2. The function is differentiable, enabling gradient-descent-based learning for modifying weights.
3. It can approximate linear and threshold activation functions by controlling weights to be very small or very large, respectively.

---

### 3.11 Feed-forward networks

Feed-forward neural networks have no feedback loops. Their output can be calculated as an explicit function of the input and weights.

---

### 3.12 Multi-layer perceptrons

A multi-layer perceptron is a feed-forward network consisting of successive layers of adaptive weights and units with threshold or sigmoidal activation functions.

The number of hidden layers dictates the network name.

---

### 3.13 MLP decision boundaries

The lecture categorizes classification problems by decision-boundary complexity:

1. Linear.
2. Convex.
3. Complex.

Corresponding perceptron architectures with threshold units:

- Single-layer perceptron: solves linear classification problems using a linear decision boundary.
- Two-layer perceptron: solves convex classification problems by intersections of linear decision boundaries.
- Three-layer perceptron: can generate arbitrary decision boundaries, including non-convex and disjoint regions.

Detailed interpretation:

- In a two-layer perceptron, each hidden unit divides the input space with a hyperplane.
- The output unit computes a logical AND of hidden-unit outputs, creating a convex decision boundary.
- In a three-layer perceptron, each unit of the second hidden layer performs a logical AND of first-hidden-layer outputs, creating a convex region.
- Different second-hidden-layer units can create different regions, possibly representing the same class.
- The output unit performs a logical OR over these regions, representing classes as sets of convex regions.

---

### 3.14 MLP universal approximation statement from the lecture

The lecture states that a three-layer perceptron with sigmoidal hidden units can approximate any smooth function to arbitrary accuracy.

The construction intuition presented:

1. A single sigmoidal hidden unit creates a smooth ridge-like response over two input variables.
2. Combining two such responses can create ridge-like functions.
3. Adding ridges can produce a function with a maximum, used as input to a second hidden layer.
4. Applying another sigmoid in the second hidden layer creates localized responses.
5. Linear combinations of localized functions in the output layer can approximate any smooth functional mapping to arbitrary accuracy.

---

### 3.15 Higher-order networks

The lecture mentions higher-order networks with the example of a one-dimensional input space containing disjoint decision regions `R_1` and `R_2`.

A linear discriminant cannot generate the required boundary, but a quadratic discriminant can.

Decision rule in the example:

```text
Assign x to C_1 if y > 0
Assign x to C_2 otherwise
```

---

### 3.16 Back-propagation

Back-propagation is presented as a learning algorithm used to modify neural-network weights. Weights are updated according to their contribution to reducing the error.

Back-propagation calculates derivatives of the error with respect to the weights.

Four conceptual steps plus update:

1. Find activations of hidden and output units after presenting an input vector.
2. Evaluate `δ` values for all output units.
3. Back-propagate `δ` values from outputs to hidden units.
4. Evaluate required derivatives as products of `δ` values and corresponding activations.
5. Update the weights.

For a hidden unit `j`:

```text
a_j = Σ_i w_ji z_i
z_j = g(a_j)
```

Derivative form:

```text
∂E_n / ∂w_ji = (∂E_n / ∂a_j)(∂a_j / ∂w_ji) = δ_j z_i
```

Output unit delta:

```text
δ_k = y_k - t_k     for each output k
```

Hidden unit delta using sigmoid derivative:

```text
δ_j = g'(a_j) Σ_k w_kj δ_k
    = z_j(1 - z_j) Σ_{k=1}^{c} w_kj δ_k
```

The name back-propagation refers to the propagation of errors backward through the network.

Online updates:

```text
Δw_ji = -η δ_j x_i
Δw_kj = -η δ_k z_j
```

Batch updates:

```text
Δw_ji = -η Σ_n δ_jn x_in
Δw_kj = -η Σ_n δ_kn z_nj
```

where `η` is the learning rate.

---

### 3.17 Gradient descent

Gradient descent is presented as a training algorithm for neural networks. Starting from an initial guess for the weight vector, the method updates weights iteratively along the negative gradient direction of the error.

Fixed-step gradient descent:

```text
Δw^t = -η ∇E |_{w^t}
```

where `w^0` is the initial guess.

Learning-rate behavior:

- Too small `η` slows convergence.
- Too large `η` causes oscillations around the solution and does not guarantee a good solution at a specific point in time.

Momentum update:

```text
Δw^t = -η ∇E |_{w^t} + μ Δw^{t-1}
```

where `μ` is the momentum constant.

---

### 3.18 Conjugate gradient

The lecture contrasts conjugate gradient with gradient descent and gradient descent with momentum.

Key point:

- Successive gradient vectors are generally not the best search directions because the new gradient is orthogonal to the previous direction.
- Conjugate gradient chooses successive search directions so that at each step, the component of the gradient parallel to the previous search direction, which has just been made zero, remains unaltered.

---

### 3.19 Minima and posterior approximation

The lecture mentions possible minima in MLP training.

It also states that network outputs approximate posterior probabilities. A visual example compares:

- Class-conditional densities used to generate data with equal priors.
- The result of training an MLP, where the solid line is the network and the dashed line is the true posterior.

---

## 4. Curriculum-Aligned Coding Guidance for Codex

### 4.1 Allowed conceptual scope

When assisting with code related to these lectures, stay within the following scope:

- Bayesian decision rules using priors, likelihoods, posteriors, risk, zero-one loss, likelihood ratios, and discriminant functions.
- Normal-density discriminant functions, including the equal spherical covariance case that produces linear decision boundaries.
- Decision trees represented as attribute tests, branches, leaves, and if-then rules.
- ID3 using entropy and information gain for categorical attributes.
- Overfitting, validation-set pruning, and the conceptual differences between ID3 and C4.5.
- Neural networks as discriminant functions.
- Linear discriminants, perceptron, logistic sigmoid, feed-forward networks, MLPs, back-propagation, gradient descent, momentum, and conjugate gradient as described above.

### 4.2 Avoid assuming unsupported methods

Do not introduce methods or implementation details not present in the lectures unless explicitly requested separately. In particular, this context file does not define or authorize additional material such as ensemble methods, random forests, boosting, CART impurity variants beyond entropy/information gain, modern deep-learning optimizers, regularization methods, dropout, batch normalization, softmax/cross-entropy derivations, convolutional networks, recurrent networks, or transformer architectures.

### 4.3 Practical implementation expectations

If implementing ID3 from this curriculum:

- Use categorical attribute-value splitting.
- Select attributes by maximum information gain.
- Compute entropy with base-2 logarithms.
- Use majority-class fallback when no attributes remain or when a branch receives no examples.
- Treat ID3 as greedy and non-backtracking.
- Discuss overfitting and pruning using the validation-set procedure if pruning is required.

If implementing neural-network components from this curriculum:

- A perceptron should use threshold output `-1/+1` and update only misclassified samples.
- Logistic sigmoid should be differentiable and bounded in `[0,1]`.
- Back-propagation should compute output deltas, propagate deltas backward, compute derivatives as delta times activation, and update weights using learning rate `η`.
- Gradient descent should follow the negative gradient of the error.
- Momentum should add `μΔw^{t-1}` to the current update.

If implementing Bayesian decision classification from this curriculum:

- Use posterior maximization for zero-one loss.
- Use conditional-risk minimization for a general loss matrix.
- For two-class risk-sensitive classification, the likelihood-ratio threshold should combine loss differences and class priors.
- For normal densities, use the discriminant form `ln p(x | ω_i) + ln P(ω_i)`.
- In the equal spherical covariance case, use linear discriminant functions derived from class means, common variance, and priors.

---

## 5. Compact Formula Index

### Bayesian decision theory

```text
P(ω_j | x) = p(x | ω_j)P(ω_j) / p(x)
p(x) = Σ_j p(x | ω_j)P(ω_j)
```

```text
P(error | x) = min[P(ω_1 | x), P(ω_2 | x)]
```

```text
R(α_i | x) = Σ_j λ(α_i | ω_j)P(ω_j | x)
```

```text
Zero-one loss: R(α_i | x) = 1 - P(ω_i | x)
```

```text
Decide ω_i if P(ω_i | x) > P(ω_j | x), ∀j ≠ i
```

```text
g_i(x) = ln p(x | ω_i) + ln P(ω_i)
```

```text
g(x) = ln[p(x | ω_1) / p(x | ω_2)] + ln[P(ω_1) / P(ω_2)]
```

```text
p(x) = 1 / ((2π)^{d/2}|Σ|^{1/2}) exp[-1/2(x - μ)^TΣ^{-1}(x - μ)]
```

```text
g_i(x) = -1/2(x - μ_i)^TΣ_i^{-1}(x - μ_i) - d/2 ln(2π) - 1/2 ln|Σ_i| + lnP(ω_i)
```

```text
Σ_i = σ²I: g_i(x) = -||x - μ_i||²/(2σ²) + lnP(ω_i)
```

```text
Σ_i = σ²I: g_i(x) = w_i^T x + w_i0
w_i = μ_i/σ²
w_i0 = -1/(2σ²) μ_i^T μ_i + lnP(ω_i)
```

### Decision trees

```text
Binary entropy: I(S) = -p_+log₂p_+ - p_-log₂p_-
```

```text
Multiclass entropy: I(S) = -Σ_i p_i log₂p_i
Maximum entropy = log₂(c)
```

```text
Gain(S,A) = I(S) - Σ_{v∈Values(A)} (|S_v|/|S|) I(S_v)
```

### Neural networks

```text
Two-class LDF: y(x) = w^T x + w_0
```

```text
Multiclass LDF: y_k(x) = W_k^T x + w_k0
```

```text
Boundary between C_k and C_j: (W_k - W_j)^T x + (w_k0 - w_j0) = 0
```

```text
Perceptron output: y = g(Σ_j w_j φ_j(x)) = g(w^Tφ)
```

```text
g(a) = -1 if a < 0; +1 if a ≥ 0
```

```text
Perceptron criterion: E_perc(w) = -Σ_{φ_n∈H} w^T(φ_n t_n)
```

```text
Logistic sigmoid: g(a) = 1/(1 + exp(-a))
```

```text
Logistic neuron: y = g(w^T x + w_0)
```

```text
Backprop output delta: δ_k = y_k - t_k
```

```text
Backprop hidden delta: δ_j = g'(a_j)Σ_k w_kjδ_k = z_j(1-z_j)Σ_k w_kjδ_k
```

```text
Derivative: ∂E_n/∂w_ji = δ_j z_i
```

```text
Online updates: Δw_ji = -ηδ_jx_i; Δw_kj = -ηδ_kz_j
```

```text
Batch updates: Δw_ji = -ηΣ_nδ_jn x_in; Δw_kj = -ηΣ_nδ_kn z_nj
```

```text
Fixed-step GD: Δw^t = -η∇E|_{w^t}
```

```text
Momentum: Δw^t = -η∇E|_{w^t} + μΔw^{t-1}
```
