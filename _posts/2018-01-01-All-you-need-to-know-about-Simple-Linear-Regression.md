---
layout: post
title:  "All you need to know about Simple Linear Regression"
date:   2018-01-01
desc: "All you need to know about Simple Linear Regression"
keywords: "Machine-Leanring,blog,Algorithm"
categories: [machine-learning]
tags: [MACHINE, LEARNING, Linear Regression]
icon: icon-html
---

Hello readers, wish you all a very happy new year!!

As we know, there are a lot of machine learning algorithms to implement, in order to solve our specific problems. But, will just knowing how to implement these, with the help of certain readymade library, save our day? Well, it might! But the truth is, just implementing them is not enough. Studying the rudiments of such algorithms not only helps in understanding these in great detail but also enhances our ability to implement them wisely as per the nature of the data.

Today, let's talk about Simple Linear Regression. What is it? What happens beneath the surface? Why should we use it? When should we use it? What kind of problems can it solve? blah? blah? blah??? If these questions resonate with yours, then you might be at the right place. So, without any wait let's dive deep into this machine learning algorithm.

Linear Regression is a supervised Machine Learning algorithm as it relies on data for training. It is one of those Machine Learning algorithms that we, the people of Computer Science, have borrowed from Statistics.  As the name suggests, it's a linear model. Linear, as in, it tries to establish a linear relationship between the dependent and the independent variable. Let's try to understand it with the help of a simple example. Let's imagine that we have information about height and weight of some people. Following is this information plotted with the help of a scatter plot.


<p align="center">
  <img title="Height vs Weight" src="/static/assets/img/posts/LinearRegressionScatterPlot.JPG">
</p>

As we can see, there seems to be a linear relationship between height and weight. By that what I mean is, weight seems to increase linearly with height. Thus, there seems to be some relationship between height and weight. And as per the plot, the best we can say, at the moment, is that the relationship seems to be linear. But, how do we represent this relationship exactly? And how would that representation help us in any way, in terms of making a prediction, if any? That's exactly what our Linear Regression algorithm will help us find out.

As we know, Linear Regression algorithm assumes that there is a linear relationship between the independent and dependent variable. (Oh! by the way, in our case weight is the dependent variable and height the independent variable. Let's just assume that from now on.) It tries to represent this linear relationship in the following form:

...
### **<center>y = b0 + (b1*x)</center>**
...

Generally,  'y' is the independent variable and 'x' is the independent variable. Linear Regression assumes that 'y' is dependent on 'x' linearly. In our case, this equation would look something like:

...
### **<center>weight = b0 + b1*height</center>**
...
 
Ok, that looks meaningful in our context now. But wait! What are these values b0 and b1? b1 is the coefficient of our dependent variable "height" and b0 is the weight-intercept. Now, Recall details about the equation of a simple line from your Geometry-knowledgebase. Don't worry if you were not able to recollect those pieces. 

I'll describe that simple concept here in short. Generally, a line in a 2-dimensional space is represented by the following equation:

...
### **<center>y = mx + b</center>**
...

, where m is the slope of the line and b is the y-intercept.
By now, you might have already gotten the main idea of Linear Regression. It'll help us find that straight line. Just that, in our case, that straight line would be represented by,

...
### **<center>weight = b0 + b1*height</center>**
...

Now, here we have our training data comprising heights and weights. In short, we have values for to put in for our "weight" and "height" variables in our equation. But, as you might have noticed, we don't know what the values of b0 and b1 are. What do we do to figure these values out? The answer is simple. We do nothing, but let our machine try to learn these values from our training data. Yes, that's what our algorithm will learn. After all, the term "Machine Learning" was not coined just randomly by some random guy playing some random sport at some random place at some random time. I guess you got it :).

But, how will our algorithm learn from the data about these values b0 and b1? Before that, let's focus on one important thing. As we know, Linear Regression algorithm tries to find a line that best describes the relationship between two variables, one is the independent one and the other dependent on that. Technically, we call that line the best-fit-line. But, how do we say that a given line is the best-fit-line? We should have some metric that would help us define the meaning of "best" in "best-fit-line". That metric is called "Ordinary Least Squares". 

Let's discuss what "Ordinary Least Squares" is. Basically, at the end of the day, we will have our algorithm predict the weights of various people based on their heights. These are called predictions. Let's denote a single prediction by y-hat-i (for the ith person). At the same time, we also have the true values of weights for the same set of values of heights. We call that true labels as y-i ( for the i-th person). Formally, we call a line the best-fit-line for which the value of "Ordinary Least Squares" (OLS) is the least. "Ordinary Least Squares" is a measure of error. It is given by:

**<center>OLS EQUATION</center>**

Now, coming back to how our Linear Regression Algorithm will learn the values of b0 and b1, it learns these value by plugging-in various values of same in our main equation at a time, which is:
weight = b0 + b1*heights

After plugging in these values, it thus calculates the predictions(weights) y-hat for all the training examples and calculates the OLS. After that, it does "something" to minimize that value of OLS. Now, what is that "something"? That "something" typically is an algorithm that comes under the category of "Optimization Algorithms". In our case, we'll use one of those algorithms called as Gradient Descent.

In Machine Learning world, there are often some functions(cost functions/objective functions) that we are interested in minimizing or as we should call it more formally, "optimising". Optimizing these functions helps us find the required values of parameters (of these functions) which yield the least values (global/local minima) for these functions when plugged-in in their respective equations. In our case, we need to optimize OLS. The parameters of the OLS are y and y-hat. y-hat depends on the values of b0 and b1. Thus, optimising OLS will help us find the required values of b0 and b1 for which we'll have the least possible value for our objective function, which is OLS. 

We'll do gradient descent to minimize this objectve function. Here's how we'll do that. We'll initialize the weights b0 and b1 with some random values. Then we'll use that set of values of b0 and b1 to predict values of weights for the entire training set. This is called as one pass or one epoch. After this, we'll calculate the gradients (slopes) of the cost function with respect to these values of b0 and b1 for the overall cost. Let's call these gradients db0 and db1. Now, we'll subtract a fraction of these gradients from our initial values b0 and b1.  In short, we'll do the following:

...
### **<center>b0 = b0 - alpha*db0</center>**
...

...
### **<center>b1 = b1 - alpha*db1</center>**
...

This will help us in descending towards the global minima, given our cost function is convex. It turns out that OLS is a conex function. Here, alpha is called the learning rate. It decides how long each step of the gradient descent will be while traveling towards the global minima. Generally, alpha should not be too large, as it results in overshooting the global minima and not too small as it takes a very long time to converge (reach global minima). A value of 0.001 or 0.0001 should be fine. But, we can experiment other values too.

So, we'll keep updating the values of b0 and b1 after each epoch which will help us move towards the global minima step by step untill the gradients (db0 and db1)assume values close to or nearly 0. And then our model/algorithm is said to have learned these values. So this is how, Linear regression works to figure out the best-fit-line.


