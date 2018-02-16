---
layout: post
title:  "Support Vector Machines"
date:   2018-02-15
desc: "demystifying Support Vector Machines"
keywords: "Machine-Leanring,blog,Algorithm,Supervised Machine Learning,Classification"
categories: [Machine-learning]
tags: [MACHINE LEARNING, Support Vector Machines, Classification]
icon: 
---

Let's talk about one of the most powerful supervised machine learning algorithms today - Support Vector Machine. Let's formally start with an introduction.
## Introduction
<p align="justify">
Support Vector Machine is a supervised machine learning algorithm which is generally used for classification tasks. It was invented by  Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1963.  This model was proposed assuming that the data points are linearly separable. Later, Vladimir N. Vapnik extended the idea of SVM and introduced something called "kernel trick" to enable non-linear classifications too.
</p>

## The Intention
<p align="justify">
How does SVM do classification tasks? We'll try to understand each and every aspect of that here. I'll not just give you a basic intuition behind SVM. Rather, my intention is to make each and everything that goes "behind the scenes" of SVM as lucid as possible. So if understanding the same is your purpose, you're at the right place. Happy reading :)
</p>

## Representation of Input Examples
<p align="justify">
As we know, in supervised machine learning, we need some training examples to train our algorithms. We generally process these examples and then come up with a function, or as we formally call that a "hypotheses", that best maps the input features to outputs/dependent values. In case of SVM, like other models, examples are represented in an n-dimensional hyperspace as vectors, where n is the number of features. For instance, if a training example, a human being, has features height, weight, and nose-length (nose-length!! Really!! That's weird, but still consider that for now :)), then this training example, which is a human being, can be represented as a vector in a 3-dimensional space with axes representing features-axes height, weight, and nose-length. Likewise,  in an n-dimensional space, all the training examples are represented as vectors where n = number of features. Basically, each example exists in the space based on their features.  
</p>

## The Objective - Binary Linear SVM Classifier
<p align="justify">
Linear SVM classifier assumes that these vectors are linearly separable.It means that we can draw a linear decision boundary that can represent our classifier. Given an n-dimensional hyperspace, this decision boundary is what we call a hyperplane. In 2-D, this hyperplane is a "line", in 3-D a "plane" and in n-dimensional hyperspace, it is called as a hyperplane. The main idea is that, whatever is the case, whether it's a line, plane or hyperplane, the decision boundary is linear in nature and it cuts the hyperspace in 2 halves in such a way that all the examples of, say class C1, lie in one half and all the others of, say class C2, lie in the other. Thus we can use that hyperplane, or decision boundary, as a classifier. So our objective is to find a hyperplane in that n-dimensional hyperspace which cuts or divides that hyperspace in such a way that all the positive training examples lie on one side of the hyperplane and negative on the other  (we generally refer the binary classes as positive and negative classes. It doesn't have to do anything with signs. Just a convention). We'll thus try to learn from our training examples the equation or representation of that hyperplane which represents our hypothesis function (H(&Theta)).
</p>

## Representation of hyperplane
<p align="justify">
Before we start with the construction of hyperplane, let's first look at how do we mathematically represent a hyperplane. As we discussed, a hyperplane in 2 -dimension is called a line. Let's see how that is represented as it's easy to plot and visualize. A line is represented by the following linear equation:
</p>


<br>
<div style="font-size: 150%; font-weight: bold; ">
<center>y = mx + b</center>
</div>
<br>

diagram line

<p align="justify">
where m is the slope of the line and b is a constant. The slope (coefficient of x) generally gives an idea of the orientation of the line with respect to x-axis and b represents the bias. Bias is a constant value that represents nothing but the position of the line.
</p>	

<p align="justify">
We can represent the same equation in the following form:
</p>

<br>
<div style="font-size: 150%; font-weight: bold; ">
<center>f(x) = W.T . X + b</center>
</div>
<br>

<p align="justify">
where W is the weight vector which contains the coefficients of x and y.
</p>

<p>
Ok, that looks meaningful in our context now. But wait! What are these values <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>? <b>&beta;<sub>1</sub></b> is the coefficient of our dependent variable "height" and <b>&beta;<sub>0</sub></b> is the weight-intercept. Now, Recall details about the equation of a simple line from your high school. Don't worry if you were not able to recollect those pieces. 
</p>

<p align="justify">
For example, for a line given by  y - 2x + 3 = 0
</p>

<p align="justify">
<center>
W.T = [1, -2], X.T = [x, y] and bias, b = 3
</center>
</p>

> The vector [1, -2] will always be normal to the line y - 2x + 3 = 0. You may try plotting things out to get better clarity.

<p align="justify">
Let's talk a bit about the representation of plane and then, maybe later, you can fuel your imagination to extend this and comprehend representation of hyperplane in n-dimensional hyperspace if you like.
</p>

<p align="justify">
In 3-D, as we discussed, this hyperplane is what we call a "plane". Similar to the representation of a line, a plane can be represented as,
</p>

<p align="justify">
<center>
aX + bY + cZ + d = 0
</center>
</p>

<p align="justify">
This equation can also be represented alternatively as,
</p>

<p align="justify">
<center>
W.T . X + B = 0
</center>
</p>

<p align="justify">
where W is the weight vector containing the coefficients of features X, Y, and Z. W is always normal to the plane and thus gives an idea about the orientation of the plane and b is the bias which represents the position of the hyperplane.
</p>

<p align="justify">
For example, for the equation of a line, 2x + 3y + 4z + 3 = 0,
</p>

<p align="justify">
<center>
W.T = [2, 3, 4], X.T = [x, y, z] and bias = b = 3
</center>
</p>

> The vector [2, 3, 4] will always be normal to the plane 2x + 3y + 4z + 3 = 0

<p align="justify">
To generalize, a hyperplane, in an n-dimensional hyperspace, can be represented by the following equation:
</p>

<p align="justify">
<center>
f(x) = W.T . X + B
</center>
</p>

<p align="justify">
where W and X are vectors of dimension 'n' which is the number of features of training examples. W is the weight vector that contains the coefficients of corresponding feature axes in X. Also, it is always normal to the given plane and thus gives an idea about the orientation of the plane in an n-dimensional hyperspace. B is the bias which describes the position of the plane in an n-dimensional hyperspace.
</p>

<p align="justify">
By now, you might have already guessed that the values of 'W' and 'b' are something that we need our algorithm to learn. But how? let's get the intuition behind building SVM model.
</p>

## Building the SVM Model - Intuition
<p align="justify">
Now that we know the representation of a hyperplane, let's discuss how exactly we find that hyperplane and build the model. But before that, let's take a look at the main idea behind building the model.
</p>

### Main Idea of the Model
#### Representation of the Model
<p align="justify">
Let's assume that the equation of the hyperplane that we need to find is as follows:
</p>


<p align="justify">
<center>
f(x) = W.T . X + b, in n-dimensional hyperspace
</center>
</p>

<p align="justify">
Now, this would divide the hyperspace into 2 halves as discussed. One will have all the positive training examples and the other, negative ones. An example is called as a positive example if it lies on the positive side of the hyperplane and negative example if it lies on the negative side of the hyperplane. Now let's assume that we pick a positive example Xi. As this is a positive example, f(Xi) should yield a value greater than 0. So,
</p>

<p align="justify">
<center>
f(Xi) = W.T . X + b > 0
</center>
</p>

<p align="justify">
Similarly, for a negative example Xj, f(Xj) should yield a value less than 0 as given below. 
</p>

<p align="justify">
<center>
f(Xi) = W.T . X + b < 0
</center>
</p>

#### Classification after learning

<p align="justify">
Once we're done with the learning, we would know the "optimized" values of the weight vector (W) and bias (b). Now during implementation/classification, we will find the value of f(Xk) for a new example 'k'. We'll then classify 'k' as a positive or negative example based on this value of f(Xk). If f(Xk) is positive, we'll classify 'k' as a positive example. If f(Xk) is negative, we'll classify 'k' as a negative example.
</p>

### Finding the Hyperplane
<p align="justify">
Now, how do we find the hyperplane? Let's consider the following 2-dimensional figure.
</p>

### diagram 1

<p align="justify">
We can see that there are many hyperplanes that separate 2 classes, represented by '*' and '0'. That looks so random. SVM doesn't choose any random hyperplane that just separates the classes. It doesn't do random stuff. Instead, SVM tries to find a hyperplane that is equidistant from the closest vectors from both the classes. These vectors that are closest to the hyperplane are called the "Supporting Vectors". Well, you got it. There's a reason behind the name of our algorithm. Now, consider the following diagram to get a better idea:
</p>

### diagram 2

<p align="justify">
We can see that the hyperplane is at distance  'd' from the supporting vectors of both the classes. SVM considers only the support vectors to construct the decision boundary-the hyperplane. It is therefore robust to outliers. Any extreme vector has absolutely zero effect on the decision boundary. SVM then tries to find this decision boundary or hyperplane by maximising the distance 'd' which represents the equal distance between itself and the supporting vectors on either of the sides.
</p>

### Let's do Some Math

<p align="justify">
Now for a positive example Xi, we can say,
</p>

<p align="justify">
<center>
f(Xi) = W.T . X + b > 0
</center>
</p>

<p align="justify">
Similarly, for a negative example Xi, we can say,
</p>

<p align="justify">
<center>
f(Xi) = W.T . X + b < 0
</center>
</p>

<p align="justify">
Now, let's introduce a variable Yi, for mathematical convenience, which represents the class of the ith example. Thus, it can have a value of either +1 or -1.
</p>

<p align="justify">
In a more general way, we can say,
</p>

<p align="justify">
<center>
Yi * (W.T . X + b) > 0
</center>
</p>

<p align="justify">
Let's investigate what we just said. Here if 'i' belong to the positive class, then Yi is positive and (W.T . X + b) is also positive. Thus their product is positive. Similarly, if 'i' belong to the negative class, then Yi is negative and (W.T . X + b) is also negative. However, the product of these still remains positive. So no matter which class is the example from, the above equation will always hold true.
</p>

<p align="justify">
Now let's look at our diagram once again.
</p>

### diagram 2

<p align="justify">
We can see that there are actually three lines. Two cross the supporting vectors and the 3rd one lies in the middle of these 2 lines. Thus, there are 3 hyperplanes when we are talking about SVM. 2 of these pass through the support vectors and the 3rd one doesn't, which is our target hyperplane that we need to find.
</p>

<p align="justify">
Now, the distance between a vector, say Xi, and our target hyperplane can be given by,
</p>

<p align="justify">
<center>
di = (W.T . X + b) / ||W||
</center>
</p>

Let's assume that the maximum margin that we desire to obtain which is feasible, is Gamma.  
Then, for all Xi, di should be greater than or equal to Gamma.

<p align="justify">
<center>
(W.T . X + b) / ||W|| >= Gamma
</center>
</p>

<p align="justify">
<center>
W.T . X + b >= Gamma * ||W||
</center>
</p>

Let's rescale Gamma * ||W|| to 1. So,

<p align="justify">
<center>
W.T . X + b >= 1, if x is a positive training example, and
</center>
</p>

<p align="justify">
<center>
W.T . X + b <= -1, if x is a negative training example
</center>
</p>

<p align="justify">
Now, let's bring in Yi again (Yi = +1 or -1, represents classes) to make our math a bit more convenient and try to represent the above two equations in just one equation.
</p>

<p align="justify">
<center>
Yi * (W.T . X + b) >= 1
</center>
</p>

Here the equality holds true only for support vectors. Thus for a support vector Xi, we'll have,

<p align="justify">
<center>
Yi * (W.T . Xi + b) = 1
</center>
</p>

<p align="justify">
We saw that W.T . Xi + b is a measure of the distance of Xi from our hyperplane (W.T . X + b) and Gamma is the distance between support vector and out hyperplane. Thus Gamma can be written as,
</p>

<p align="justify">
<center>
Gamma = (W.T . X + b) / ||W||
</center>
</p>

<p align="justify">
Now, we want to maximize Gamma. You can now say that the minimum value of ||W|| is 0. Yes, you are right. The minimum is 0. BUT, that is true when we don't have any constraints. But, if you look closely, Gamma is the distance between our hyperplane and support vectors. And for support vectors, we have,
</p>

<p align="justify">
<center>
Yi * (W.T . Xi + b) = 1
</center>
</p>


<p align="justify">
This is our constraint. So we need to minimize ||W|| subject to the constraint Yi * (W.T . Xi + b) = 1. Thus, we need to do a constrained optimization. That seems difficult. Well, thanks to Joseph-Louis Lagrange, we can very well do that with ease.
</p>

### Time to thank Joseph-Louis Lagrange

<p align="justify">
Lagrange taught us how to minimize a function subject to some constraints. He introduced something called as Lagrange-Multipliers. With the help of those, we can very well represent a function, without any constraints, that is a combination of the function we want to optimize and its constraint. We call is function Lagrangian. So, if f(x) is a function that we need to optimize and g(x) is the constraint we need to obey while optimizing, then the Lagrangian L(x, lambda) can be written as:
</p>

<p align="justify">
<center>
L(x, lambda) = f(x) - lambda * g(x)
</center>
</p>





















