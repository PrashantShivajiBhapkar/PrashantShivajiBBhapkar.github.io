---
layout: post
title:  "Demystifying Support Vector Machines"
date:   2018-03-04
desc: "All you need to know about Support Vector Machines"
keywords: "Machine-Leanring,blog,Algorithm"
categories: [Machine-learning]
tags: [MACHINE LEARNING, Support Vector Machines, Linear Classification, SVM]
icon: 
---

# Demystifying Support Vector Machines

Let’s talk about one of the most powerful supervised machine learning
algorithms-Support Vector Machine or SVM. Let’s formally start with an
introduction.

### Introduction

Support Vector Machine or **SVM** is a supervised machine learning algorithm
which is generally used for classification tasks. The idea was presented by
Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1963 assuming that the data
points are linearly separable. Later, Vladimir N. Vapnik extended this initial
idea and introduced something called as **“kernel trick”** to enable non-linear
classifications too.

### The Intention

How does SVM do classification tasks? We’ll try to understand each and every
aspect of that here. I’ll not just give you a basic intuition behind SVM.
Rather, my intention is to make each and everything, that goes *“behind the
scenes”* of SVM, as lucid as possible. So if understanding the same is your
purpose, you’re at the right place. Happy reading :)

#### **Representation of input examples**

As we know, in supervised machine learning, we need some training examples to
train our algorithm on. We generally process these examples and then come up
with a function, or as we formally call a *“***hypothesis***”*, that best maps
the input features *(independent variables)* to dependent values. In case of
SVM, like other models, examples are represented in an n-dimensional hyperspace
as vectors, where n is the number of features. For instance, if a training
example, a human being for instance, has features height, weight, and
nose-length (nose-length!! Really!! That’s weird, but still consider that for
now :)), then this training example, which is a human being, can be represented
as a vector in a 3-dimensional space with axes representing features-height,
weight, and nose-length. Likewise, examples having **n** features can be
represented in an **n**-dimensional hyperspace as vectors. Basically, each
example exists in the space where it’s location is determined by the values of
it’s features.

### The Objective — Binary Linear SVM Classifier

Linear SVM classifier assumes that these vectors are linearly separable.It means
that we can draw a linear decision boundary that can represent our classifier.
Given an n-dimensional hyperspace, this decision boundary is what we call a
hyper-plane. In 2-D, this hyper-plane is a “line”, in 3-D a “plane” and in
n-dimensional hyperspace, it is called as a hyper-plane. The main idea or
assumption is that, whatever is the case *(1-D, 2-D or n-D)*, whether it’s a
line, plane or hyper-plane, the decision boundary is linear in nature and it
cuts the hyperspace in 2 halves in such a way that all the examples of, say
class **C1**, lie in one half and all the others of, say class **C2**, lie in
the other. Thus, we can use that hyper-plane, or decision boundary, as a
classifier. So our objective is to find a hyper-plane in that n-dimensional
hyperspace which cuts or divides that hyperspace in such a way that all the
positive training examples lie on one side of it and negative on the other *(we
generally refer the binary classes as positive and negative classes. It
***doesn’t ***have to do anything with signs. Just a convention)*. We’ll thus
try to learn from our training examples the equation or representation of this
hyper-plane which represents our hypothesis *(In machine learning world, we
always make some hypothesis of the function that actually exists between input
and output and proceed)*.

### **Representation of a hyper-plane**

Before we start with the construction of our hyper- plane, let’s first look at
how do we mathematically represent a hyper- plane. As we discussed, a hyper-
plane in a 2-dimensional space is called a line. Let’s see how that is
represented as it’s easy to plot and visualize. A line is represented by the
following linear equation:<br> **y = mx + b,**

<span class="figcaption_hack">Representation of a line in a 2 dimensional space</span>

where m is the slope of the line and b is a constant. The slope (coefficient of
x) generally gives an idea of the orientation of the line with respect to x-axis
and b represents the bias. Bias is a constant value that represents nothing but
the position of the line.

We can represent the same equation in the following form:<br> **f(x) =
W.Transpose . X + B,**

where **W** is the **weight** vector which contains the coefficients of **x** and **y**.<br> For example, for a line given by x-2y+3 = 0,<br> **W** = [1,
-2], **X** = [x, y] and bias **b** = 3

> The vector [1, -2] will always be normal to the line y — 2x + 3 = 0. You may try
> plotting things out to get a better clarity

Let’s talk a bit about the representation of plane now, and then, maybe later,
you can fuel your imagination to extend this and comprehend representation of
hyper-plane in an n-dimensional hyperspace.<br> In a 3-Dimensional space or
hyperspace, as we discussed, this hyper-plane is what we call a “plane”. Similar
to the representation of a line, a plane, in a 3-dimensional space can be
represented as,<br> **aX + bY + cZ + d = 0**

This equation can also be represented alternatively as,<br> **W.Transpose . X +
B = 0,**

where **W** is the **weight** vector containing the coefficients of features
**X**, **Y**, and **Z**. **W** is always normal to the plane and thus gives an
idea about the orientation of the plane and b is the bias which represents the
position of the hyper-plane.<br> For example, for the equation of a line, 2x +
3y + 4z + 3 = 0,<br> W = [2, 3, 4], X = [x, y, z] and bias, B= 3

> The vector [2, 3, 4] will always be normal to the plane 2x + 3y + 4z + 3 = 0

To generalize, a hyperplane, in an n-dimensional hyperspace, can be represented
by the following equation:<br> **f(x) = W.Transpose . X + B,**

where **W **and **X** are vectors of dimension ’n’ which is the number of
features of training examples. **W** is the **weight** vector that contains the
coefficients of corresponding feature axes in **X**. Also, it is always normal
to the given plane and thus gives an idea about the orientation of the plane in
an n-dimensional hyperspace. **B** is the **bias** which describes the position
of the plane in an n-dimensional hyperspace.<br> By now, you might have already
guessed that the values of ‘**W**’ and ‘**B**’ are something that we need our
algorithm to learn. But how? let’s get the intuition behind building SVM model.

### Building the SVM Model — Intuition

Now that we know the representation of a hyper-plane, let’s discuss how exactly
we find that hyper-plane and build the model. But before that, let’s take a look
at the main idea behind building the model.

### **Main Idea of the Model**

**Representation of the Model**<br> Let’s assume that the equation of the
hyper-plane that we need to find is as follows:<br> **f(x) = W.Transpose . X +
b,** in n-dimensional hyperspace

Now, this would divide the hyperspace into 2 halves as discussed. One will have
all the positive training examples and the other, negative ones. An example is
called as a positive example if it lies on the positive side of the hyper-plane
and negative example if it lies on the negative side of the hyper-plane.

Now let’s assume that we pick a positive example Xi. As this is a positive
example, **f(Xi)** should yield a value greater than 0. So,<br> **f(Xi) =
W.Transpose . X + b > 0**

Similarly, for a negative example **Xj**, **f(Xj)** should yield a value less
than 0 as given below. <br> **f(Xj) = W.Transpose . X + b < 0**

**Classification after learning**<br> Once we’re done with the learning, we
would know the “**optimized**” values of the **weight** vector (**W**) and
**bias** (**b**) for which the error or loss would be minimum. Now during
implementation/classification, while trying to classify a new example **‘Xk’,
**we will find the value of **f(Xk)**. We’ll then classify** ‘Xk’ **as a
positive or negative example based on this value of **f(Xk)**. If **f(Xk)** is
**positive**, we’ll classify **‘k’** as a **positive **example. If **f(Xk)** is
**negative**, we’ll classify ‘**k**’ as a **negative **example.

**Finding the Hyper-plane**<br> Now, how do we find the hyper-plane? Let’s
consider the following 2-dimensional figure.

<span class="figcaption_hack">Examples in a 2-dimensional space with 2 features</span>

We can see that there are many hyper-planes *(or lines)* that separate 2
classes, represented by red and yellow color. All these lines look so random and
which one would you choose as your classifier? SVM doesn’t choose any random
hyper-plane that just separates the classes. It doesn’t do random stuff.
Instead, SVM tries to find a hyper-plane *(or a line in this case) *that is
equidistant from the closest vectors from both the classes. These vectors that
are closest to the hyper-plane are called as “Supporting Vectors”. Well, you got
it. There’s a reason behind the name of our algorithm.

Now, consider the following diagram to get a clearer idea:

<span class="figcaption_hack">Examples of 2 classes separated with the help of supporting vectors</span>

We can see that the hyper-plane is at distance ‘d’ from the supporting vectors
of both the classes. SVM considers only the supporting vectors or support
vectors to construct the decision boundary *(the hyper-plane)*. It is therefore
robust to outliers. Any vector present far away from the supporting vectors has
absolutely zero effect on the decision boundary. SVM then tries to find this
decision boundary or hyper-plane by maximizing the distance or margin ‘**d**’
which represents the equal distance between itself and the supporting vectors on
either of the sides.

### **Time For Some Math**

Now for a positive example **Xi**, we can say,<br> ** f(Xi) = W.Transpose . X +
b > 0**

Similarly, for a negative example **Xj**, we can say,<br> ** f(Xj) = W.Transpose
. X + b < 0**

Now, let’s introduce a variable **Yi**, for mathematical convenience, which
represents the class of the* *example **i**. Thus, it can have a value of either
**+1** or **-1**.

In a more general way, we can say,<br> **Yi* (W.Transpose . X + b) > 0**

Let’s investigate what we just said. Here if **‘i’** belongs to the positive
class, then **Yi **is positive and **(W.Transpose . X + b)** is also positive.
Thus their product is positive. Similarly, if **‘i’** belong to the negative
class, then **Yi** is negative and **(W.Transpose . X + b)** is also negative.
However, the product of these still remains positive. So no matter which class
is the example from, the above equation will always hold true.

Now let’s look at our diagram once again.

![](https://cdn-images-1.medium.com/max/720/1*HLpdTmpTj1u7GF4tlfBvAg.png)
<span class="figcaption_hack">Examples of 2 classes separated with the help of supporting vectors</span>

We can see that there are actually three lines. Two cross the supporting vectors
and the 3rd one lies exactly in the middle of these 2 lines. Thus, there are 3
hyperplanes when we are talking about SVM. 2 of these pass through the support
vectors and the 3rd one doesn’t, which is our target hyperplane that we need to
find.

> **GIST: In SVM learning, we want the margin or distance (d) between the
> hyper-plane and support vectors to be as large as possible. WHY? Just try to
think what we are trying to achieve here and how will ‘having maximum
margin/distance’ help in that. So, more the distance, better the classifier is.
We don’t want any of the example to fall in the gutter (the space between
hyper-planes that pass through support vectors). We need to respect this
condition when we try to maximize the margin mathematically. This condition, is
referred to as a “constraint”, in mathematical parlance.**

Now, the distance between a vector, say **Xi**, and our target hyper-plane can
be given by,

**di = (W.Transpose . X + b) / ||W||**

Let’s assume that the **maximum margin** that we desire to obtain, which is
feasible, is **Gamma**. Then, for all **Xi**, **di **should be greater than or
equal to **Gamma**.

**(W.Transpose . X + b) / ||W|| >= Gamma**

**=> W.Transpose . X + b >= Gamma * ||W||**

Now, let’s re scale **Gamma * ||W||** to **1**. So,

**W.Transpose . X + b >= 1**, if **x** is a **positive **training example,
and<br> **W.Transpose . X + b <= -1**, if **x** is a **negative **training
example

Now, let’s bring in **Yi ***(Yi = +1 or -1, represents classes) *to make our
math a bit more convenient and try to represent the above two equations in just
one equation.

**Yi * (W.Transpose . X + b) >= 1 ***(oh Math! you’re so beautiful :))*

Here the equality holds true only for support vectors. Thus for a support vector
**Xi**, we’ll have,

**Yi * (W.Transpose . Xi + b) = 1**

We saw that “**W.Transpose . Xi + b”** is a measure of the distance of **Xi
**from our hyper-plane **(W.Transpose . X + b)** and **Gamma **is the distance
between the support vector(s) and our hyper-plane. Thus **Gamma **(for support
vectors obviously) can be written as,

**Gamma = (W.Transpose . X + b) / ||W||**

Now, we want to maximize **Gamma**. You can now say that the maximum value of
**Gamma** is **infinity **as** **the minimum value of **||W||** is **0**. Yes,
you are right. The maximum possible value of **Gamma** is **infinity**. **BUT**,
that is true when we **don’t **have any constraints. If you look closely,
**Gamma **is the distance between our hyper-plane and support vectors. And for
support vectors, the following condition **MUST **hold true,

**Yi * (W.Transpose . Xi + b) = 1**

This is our **constraint**. So we need to minimize **||W||** **SUBJECT** to the
constraint **Yi * (W.Transpose . Xi + b) = 1**. Hence, we need to do a
constrained optimization here. Well, thanks to Joseph-Louis Lagrange, we can
very well do that with ease. Now, once we are done with this optimization, we’ll
get the optimal values for our **weight** vector **W** and **bias b.**

> **And that’s all we wanted!! That’s the end of learning phase of our algorithm.
> Our model/machine would be said to have learned something based on
something.What are these ‘some-things’. I guess you know now.**

So reader, this marks the end of your journey with me in learning SVM where now
I pass the baton to you and hope that you complete your final, remaining short
part of the journey with no one other than **Mr. Joseph-Louis Lagrange**. Read
about how we do constrained optimizations with **Lagrange multipliers** and
you’ll solve the final piece of this puzzle. It’s not that difficult. Just watch
some [Khan Academy
Videos](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/constrained-optimization/a/lagrange-multipliers-single-constraint)
about those and you’ll get the idea.

### CONCLUSION

Ouch!! Too much of theory and math. That might have been quite a heavy dose for
those readers reading about SVM for the very first time. Yes I admit the fact
that the subject looks a bit difficult to comprehend in the very first exposure
to it. However, I’ve tried to be as lucid as possible in explaining the relevant
concepts keeping in mind the first flyers especially. Finally, believe me,
**you’ll learn everything you want**. Just be a bit patient and keep on reading
from one source or the other about the concept you want to learn. I hope this
post helped in clarifying stuff pertaining to SVM and that this was not a
perplexing explanation. Was it? or was it rather boring? Let me know in the
comments section below and I’ll consider your points to focus on for the next
time. Till then, goodbye and thanks for your time. Have a good time :).