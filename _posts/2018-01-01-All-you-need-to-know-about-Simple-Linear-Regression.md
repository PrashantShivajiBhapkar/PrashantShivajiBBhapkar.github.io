---
layout: post
title:  "All you need to know about Simple Linear Regression"
date:   2018-01-01
desc: "All you need to know about Simple Linear Regression"
keywords: "Machine-Leanring,blog,Algorithm"
categories: [Machine-learning]
tags: [MACHINE, LEARNING, Linear Regression]
icon: icon-html
---

Hello readers, wish you all a very happy new year!!
<p align="justify">
As we know, there are a lot of machine learning algorithms to implement, in order to solve our data-centric problems. But, will just knowing how to implement these, with the help of certain readymade library, save our day? Well, it might! But the truth is, just implementing them is not enough. Studying the rudiments of such algorithms is also important. It will not only help us in understanding these in great detail, but also enable us in implementing these wisely.
</p>

<p align="justify">
Today, let's talk about Simple Linear Regression. What is it? What happens beneath the surface? Why should we use it? When should we use it? What kind of problems can it solve? blah? blah? blah??? If these questions resonate with yours, then you might be at the right place. So, without any wait let's dive deep into this machine learning algorithm.
</p>

## What is Simple Linear Regression?
<p align="justify">
Simple Linear Regression is a supervised Machine Learning algorithm as it relies on data for training. It is one of those Machine Learning algorithms that we have borrowed from Statistics.  As the name suggests, it's a linear model. Linear, as in, it tries to establish a linear relationship between the dependent and the independent variable. Let's try to understand it with the help of a simple example. Let's imagine that we have information about heights and weights of some people. Following is this information plotted on a scatter plot.
</p>

<br>
<p align="center">
  <img alt="Detailed wallpaper collection item screenshot" title="Height vs Weight"  src="/static/assets/img/posts/LinearRegressionScatterPlot.JPG">
</p>
<br>

<p align="justify">
As we can see, there seems to be a linear relationship between height and weight. By that what I mean is, weight seems to increase linearly with height. As per the plot, the best we can say, at the moment, is that the relationship seems to be linear. But, how do we represent this relationship exactly? And how would that representation help us in any way, in terms of making a prediction, if any? That's exactly what our Linear Regression algorithm will help us find out.
</p>

<p align="justify">
As we know, Linear Regression algorithm assumes that there is a linear relationship between the independent and dependent variable. (Oh! by the way, in our case weight is the dependent variable and height the independent variable. Let's just assume that from now on.) It tries to represent this linear relationship in the following form:
</p>

<br>
<div style="font-size: 150%; font-weight: bold; ">
<center>y = <b>&beta;<sub>0</sub></b> + (&beta;<sub>1</sub>*x)</center>
</div>
<br>

<p align="justify">
In mathematical parlance, we say that y is a function of x. In other words, it simply means that the value of 'y' is dependent on value of 'x'. In the scope of Linear Regression, 'y' is dependent on 'x' linearly. Now, in our case, this equation would look something like this:
</p>	

<br>
<div style="font-size: 150%; font-weight: bold; ">
<center>weight = <b>&beta;<sub>0</sub></b> + &beta;<sub>1</sub>*height</center>
</div>
<br>

<p>
Ok, that looks meaningful in our context now. But wait! What are these values <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>? <b>&beta;<sub>1</sub></b> is the coefficient of our dependent variable "height" and <b>&beta;<sub>0</sub></b> is the weight-intercept. Now, Recall details about the equation of a simple line from your high school. Don't worry if you were not able to recollect those pieces. 
</p>

<p align="justify">
I'll describe that simple concept here in short. Generally, a line in a 2-dimensional space is represented by the following equation:
</p>

<br>
<div style="font-size: 150%; font-weight: bold; ">
<center>y = mx + b</center>
</div>
<br>

<p align="justify">
, where m is the slope of the line and b is the y-intercept.
By now, you might have already gotten the main idea of Linear Regression. It'll help us find that straight line. Just that, in our case, that straight line would be represented by: <b>weight = <b>&beta;<sub>0</sub></b> + &beta;<sub>1</sub>*height</b>
</p>

<p align="justify">
Now, here we have our training data comprising heights and weights. In short, we have values for to put in for our "weight" and "height" variables in our equation. But, as you might have noticed, we don't know what the values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b> are. What do we do to figure these values out? The answer is simple. We do nothing, but let our machine try to learn these values from our training data. Yes, that's what our algorithm will learn. After all, the term "Machine Learning" was not coined just randomly by some random guy playing some random sport at some random place at some random time. I guess you got it :).
</p>

## How does the Machine actually learn?
<p align="justify">
How will our algorithm learn from the data about these values <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>? Let's focus on one important thing now. As we know, Linear Regression algorithm tries to find a line that best describes the relationship between the independent and dependent variables. Technically, we call that line the best-fit-line. But, how do we say that a given line is the best-fit-line? We should have some metric that would help us define the meaning of "best" in "best-fit-line". That metric is called "Ordinary Least Squares". 
</p>
<p align="justify">
Let's discuss what "Ordinary Least Squares" is. Basically, at the end of the day, we will have our algorithm predict the weights of various people based on their heights. These are called predictions. Let's denote a single prediction by y_pred<sup>i</sup> (for the ith person). At the same time, we also have the true values of weights for the same set of people. We call those true labels as y<sup>i</sup> ( for the i-th person). We take all the predictions of weights y_pred<sup>i</sup> and true labels y<sup>i</sup> over our entire training data. We then find the difference between each y_pred<sup>i</sup> and y<sup>i</sup> pair and then add all such differences together. The best fit line is that line for which the value of this metric (OLS) is the least. Thus, "Ordinary Least Squares" gives us a measure of how well our predictions for values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>. We formally call functions like these in Machine Learning as cost functions. The equation of OLS can be represented as:
</p>

<p align="center">
  <img alt="Detailed wallpaper collection item screenshot" title="Cost Function"  src="/static/assets/img/posts/LinearRegressionCostFunction.JPG">
</p>

<p>
where h<sub>θ</sub>(X<sup>(i)</sup>) is our Hypothesis which is nothin but <i><b>h<sub>θ</sub>(X<sup>(i)</sup>)</b></i> or weight or <i><b>y_pred<sup>i</sup> = &beta;<sub>0</sub> + &beta;<sub>1</sub></b></i>, <i><b>y<sup>i</sup></b></i> is the true label and <i>m</i> is the number of training examples.
</p>
<p align="justify">
But, the question - "How Linear Regression Algorithm will learn the values of <b>&beta;<sub>0</sub></b> and &beta;<sub>1</sub>?" still remains unanswered. It turns out that it learns these value by plugging-in various values of same in our main equation at a time, which is <b>weight = <b>&beta;<sub>0</sub></b> + &beta;<sub>1</sub>*heights</b>. After plugging in these values, it thus calculates the predictions(weights) y_pred<sup>i</sup> for all the training examples and calculates the OLS. After that, it does "something" to minimize that value of OLS. Now, what is that "something"? That "something" typically is an algorithm that comes under the category of "Optimization Algorithms". In our case, we'll use one of those algorithms called as Gradient Descent.
</p>
<p align="justify">
In Machine Learning world, there are often some functions(cost functions/objective functions) that we are interested in minimizing or as we should call it more formally, "optimising". Optimizing these functions helps us find the required values of parameters (of these functions) which yield the least values (global/local minima) for these functions when plugged-in in their respective equations. In our case, we need to optimize OLS. The parameters of the OLS are y and y_pred<sup>i</sup>. y_pred<sup>i</sup> depends on the values of <b>&beta;<sub>0</sub></b> and &beta;<sub>1</sub>. Thus, optimising OLS will help us find the required values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b> for which we'll have the least possible value for our objective function, which is OLS. 
</p>

## Gradient Descent - a quick overview
<p align="justify">
We'll do gradient descent to minimize this objectve function. Here's how we'll do that. We'll initialize the values of  <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b> with some random values. Then we'll use these to predict values of weights for the entire training set. This is called as one pass or one epoch. After this, we'll calculate the gradients (slopes) of the cost function for these values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>. Let's call these gradients d<b>&beta;<sub>0</sub></b> and d&beta;<sub>1</sub>. Now, we'll subtract a fraction of these gradients from our initial values <b>&beta;<sub>0</sub></b> and &beta;<sub>1</sub>.  In short, we'll do the following:
</p>

<br>
<div style="font-size: 100%; font-weight: bold; ">
<center>&beta;<sub>0</sub> = <b>&beta;<sub>0</sub></b> - <b>&alpha</b>;*d&beta;<sub>0</sub></center>
</div>
<div style="font-size: 100%; font-weight: bold; ">	
<center>&beta;<sub>1</sub> = <b>&beta;<sub>1</sub></b> - <b>&alpha</b>;*d&beta;<sub>1</sub></center>
</div>
<br>

<p align="justify">
This will help us in descending towards the global minima, given our cost function is convex. It turns out that OLS is a convex function. Here, <b>&alpha</b>; is called the learning rate. It decides how long each step of the gradient descent will be while traveling towards the global minima. Generally, <b>&alpha</b>; should not be too large, as it results in overshooting the global minima and not too small as it takes a very long time to converge (reach global minima). A value of 0.001 or 0.0001 should be fine. But, we can experiment other values too.
</p>

<p align="justify">
Thus, we'll keep updating the values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b> after each epoch. This will help us move towards the global minima step by step untill the gradients (d<b>&beta;<sub>0</sub></b> and d&beta;<sub>1</sub>) assume values close to or nearly 0. And then our model/algorithm is said to have learned these values.
</p>

## Ok, what's there after learning values of &beta;<sub>0</sub> and &beta;<sub>1</sub>
<p>
Well, once our model has learned the values of <b>&beta;<sub>0</sub></b> and <b>&beta;<sub>1</sub></b>m we can use it to make predictions about dependent variable given we have information about independent variable. In our case, we can predict the weight of a person if we know their height. In real life, businesses might be least interested in predicting weight of someone. But, they might still want to predict something. It might be something that might have impact on their revenue. It can be anything. But most importantly, if that "something" is dependent linearly only on one variable, then now we better know what to do.
</p>


## Conclusion
<p>So, now we learned that Simple Linear Regression is a Supervised Machine Learning algorithm which is used to describe a linear relationship between the independent and dependent variable. Unfortunately it cannot work on more than 2 variables. For that, we have another algorithm which is called as Multiple Linear Regression. Also, I apologize for not going into great detail while explaining gradient descent. It could've consumed a lot of space :).  Anyway, I've tried to give all the necessary details of all the relevant concepts at appropriate depth. Please feel free to comment down as to how you found this post. Please feel free to share if you liked it. Thanks. :)</p>

#### You may check the PYTHON implementation of how to code Linear Regression from scratch [here](https://github.com/PrashantShivajiBhapkar/Machine-Learning-Algorithms-from-Scratch/blob/master/LinearRegression.py)