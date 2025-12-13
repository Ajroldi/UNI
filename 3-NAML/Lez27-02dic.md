00:00:01 Speaker 1
Last time, just a brief recap, last time we have introduced the problem of trying to understand the ability if a neural network is able to represent or, more precisely, to approximate any function. In that lecture, we have given what we can call a visual proof that this is true.

00:00:33 Speaker 1
We have used some tricks, starting either from a sigmoidal function or a real function, and in both cases, we have essentially understood that, The idea is to, at least from the visual point of view, is to construct a sort of, let me say, basis for the particular, for a particular space.

00:01:06 Speaker 1
In the first case, we have used the basis, the column function. In the second, with the ReLU, we have the head function. And then, by exploiting these constructed basis, we can actually represent any function with arbitrary accuracy, simply by using the linear combination of those functions. And the accuracy is given by the number of basis functions that we are using.

00:01:42 Speaker 1
Okay, so this is, if you want, we have... somehow recast the problem into an approximation problem and interpolation theorem, okay. For the theoretical proof, the path is a little bit different, and in particular, what we are going to see is actually the proof just for shallow network.

00:02:15 Speaker 1
So network characterized by only one hidden layer. The proof for more general networks, so deep networks, is much more complicated, but it's important to grasp the idea in the shallow case, which is enough to say that actually with neural network we can approximate any function, okay. In order to understand at least the ideas, the main ideas of the proof, we need some concepts from functional analysis.

00:02:52 Speaker 1
So today I'm going to start from very basic concepts. Maybe some of these concepts will be also annoying because we have already encountered some of these concepts during the course. But I have just tried to summarize everything into this presentation in order for you to have everything in one place if you want to study or recall some concepts.

00:03:25 Speaker 1
So here is a brief description of what we want to cover. So why functional analysis? Well, actually, the motivations, at least for the neural network, has been just mentioned. In particular, the idea is that we, in deep learning or machine learning in general, we are using finite dimensional vectors, the feature vectors is characterized by a finite dimension.

00:04:05 Speaker 1
So, the idea is that here we want to try to understand how we can extend the concepts like length, angle and convergence from Rn, which are usual spaces, to infinite dimensional spaces. The Sibenko theorem that is mentioned here is actually the first one.

00:04:36 Speaker 1
The first form that has been proposed, the first... result about the universal approximation theorem for shallow networks. After this result, many other results have been proposed, starting from that one, but the ideas are the same, with just some technicalities or generalization, so mainly what we will consider is this theorem, and if you want also to take a look at the paper, the original.

00:05:14 Speaker 1
paper, it's not very long, it's 15 pages, and it's quite clear, so you can take a look at it. Okay, so let's start with the basic concept, vector space. A real vector space, we have already started in the first lecture by considering vector spaces. It is a set with some operations, and for these operations we have some properties that are listed there.

00:05:59 Speaker 1
And in general, we can say that why these vector spaces are important, because they are the most general. Actually, there is a concept which is even more general than vector spaces, which is a topological space. But at least for practical purposes, we can start with vector spaces. They are the suitable framework for dealing with functions and infinite-dimensional options.

00:06:34 Speaker 1
So, this is the definition. So, it's actually here... Yeah, okay, I thought there was a mistake. So, here we have some examples. Rm is a vector space. The polynomial of degree less than or equal to k on an interval.

00:07:06 Speaker 1
The continuous function on an interval. And... I guess that many of you are familiar with L2 space, a square integrable function, on a generic domain omega. We will come back to this last space later. Then we have the concept of basis. The basis is the minimum set of linearly independent functions that allows to represent any other vector of the space as a linear combination of those functions.

00:07:43 Speaker 1
This is an important concept because, for example, in finite elements this concept is paramount important because once you have chosen the type of finite element that you want to use, P1, P2, P3, Q1 or other exotic finite elements, what you have to do is to decide what basis you want to use for that particular space. So it has to be finite. This is the next function or other function for higher order polynomial.

00:08:18 Speaker 1
And c1 up to cn are the constants that allow us to represent any function. Essentially, if you remember what we have done last time graphically, you remember that there was a last slider in the animation that I have shown, which was able to give the height of the function. Essentially, this height is given by this represent. Assuming that the basis functions are the maximum height of one, which is something common.

00:08:59 Speaker 1
Obviously, depending on the space, these... Functions or vectors are different object for our end. These fee 1 up to pn are vectors of canonical vectors. 0, 0, 1 is so, um, then once you have a vector space, uh, you can introduce, uh, uh, to other, uh, important, uh, um, operations, which are, uh, the inner product.

00:09:37 Speaker 1
First, and then the norm, uh, the inner product is, uh, uh, a function that takes, uh, two elements of the space and it returns a number, um, which, uh, is assumed to be either positive, uh, or, um, Equal to zero if evaluated on the same argument, and it is equal to zero only when u is equal to zero, okay?

00:10:13 Speaker 1
Then if the arguments are different, it's symmetric and it's linear. With respect, actually, to both arguments, you can also have linearity on the second argument. Why it is important? Because in general, in RN we are very familiar with this concept, but in general it allows to measure angles and orthogonality.

00:10:47 Speaker 1
So, for example, when we have dealt with orthogonal matrices, the idea was to come up with, for example, starting from a given matrix and orthogonalizing this matrix amounts to find a matrix in which the same column space is represented by vectors, which are unitary and orthogonal one to the other.

00:11:20 Speaker 1
And in general, orthogonality can be checked also when the elements of the basis, or in general the function, the the phi are not only vectors, but also generic functions. Here we have some examples of the inner products. In Rn, it's usual in a product that we know very well.

00:11:53 Speaker 1
For functions, in the interval is actually the continuous counterpart of the previous, so it's the integral between 0 and 1, and in C1 we have not only the values of the product of the values of the functions, but also the product of the first derivative. This is the inner product in C1.

00:12:26 Speaker 1
Once you have an inner product, You have, for sure, always a norm, which can be defined as the square root of the inner product between you and itself. Remember that this is positive, or zero, so this operation makes sense. Moreover, we know that a norm is an operation that takes a vector and returns a number.

00:13:10 Speaker 1
And this number is either positive or zero, and it is zero only when the vector is the null vector or the null element in general, which is coherent with what we have seen for the inner product. And it is homogeneous, and this is a direct consequence of the linearity that we have seen before. And moreover, we have the triangle inequality.

00:13:45 Speaker 1
It's clear that the concept of norm is a generalization of the concept of length or distance that we know very well and we are used to in the space Rn to any vector space. And this is of paramount importance because norm is used for checking convergence and approximation. So when you have, and we have seen this fact many times, when we have, for example, in an iterative process, we have to devise a stopping criterion.

00:14:28 Speaker 1
We have to measure the norm between, for example, two consecutive iterates. Or the norm of the residual. Anyhow, we have to measure something that is a, related to the convergence of the algorithm or more in general we will see that convergence here we with the concept of convergence we mean also.

00:14:58 Speaker 1
something different and approximation approximation is for example related to what we have seen last time when we have a function and we want to approximate that function by for example a linear combination of basis function like infinite elements infinite elements you can devise what it's called a finite.

00:15:29 Speaker 1
element error estimates which is nothing more than a measure of the error that you are committing by approximating the true solution of a PDE for example with a finite element one and this is measured in some norm and this is measured in some norm, And usually on the right-hand side of the, so you have, let me say, U minus UH, where U is the true solution and UH is the approximate solution, is less than or equal to some constant.

00:16:04 Speaker 1
And then you have the discretization parameter that is related to how many functions you are using for, a basis function you are using for approximating the original solution of the PDE. And then here you have an exponent, which is giving you the order of convergence of the method. And,

00:16:34 Speaker 1
and there are some other things here we are not going to discuss. In this context, but the important thing is that here we are the norm that is used to measure the approximation error. The common norms for vectors are the ones that we have already seen. So Euclidean maximum norm, one norm or another norm.

00:17:09 Speaker 1
And for functions, we have, if you want, the continuous counterpart. So we have the L2 norm, which is the integral over the given interval of the function of the absolute value of the function squared. And then you take the square root. And obviously, we are seeing this concept already for matrices, for example, where we have introduced the two metrics, the provenience metrics, provenience norms, sorry.

00:17:51 Speaker 1
And essentially, different norms are giving you different measures of smallness or, in general, the different measures of a given object. And how to choose which norm to use depends on the context. So, for example, here, the usual.

00:18:21 Speaker 1
The usual norm that you adopt here is the one norm, actually the H1 norm, but you can also, if. You are just interested in controlling the functions and its gradients, its gradient, or you can use also H2 or HP norm if you want to have a higher control on the behavior of the function.

00:18:55 Speaker 1
So, the... For, there is a relationship between inner product and norm, actually this is the so-called Cauchy-Schwarz inequality. Essentially, you can say that the absolute value of the inner product between U and D is equal to the product of the norms of the two objects.

00:19:31 Speaker 1
Essentially, this is a generalization of what we know very well that for usual vectors, we have that the scalar product between one possible way of defining the scalar product between two vectors, is the product of the norm times the cosines of the angle. between the two vectors, so if you want.

00:20:06 Speaker 1
what we have over there is the generalization of this concept. Here is an example of application of the Cauchy-Schwarz inequality for the C0 norm of two functions. So you have here the scalar product in C0 and here you have the two norms of the functions. Now comes an important.

00:20:47 Speaker 1
point, the point of convergence and completeness of space. Here we have to define two concepts. The first one is the concept of Cauchy sequence, here you have the definition, essentially if you have a sequence, here it's Vi, this is called a Cauchy sequence if you can pick any epsilon greater than zero,

00:21:22 Speaker 1
and there is always an index n, such that for any two indices, i and j, greater than n, the norm of the difference between Vi and Vj, which are two terms of the sequence with index greater than n, is smaller than epsilon.

00:21:55 Speaker 1
The convergence, concept, is slightly different. It says that given a sequence and an epsilon greater than zero, the sequence is said to be convergent if the norm of the difference between the i and v, which is the limit value of the sequence, is smaller than or equal to epsilon for all i greater than a suitable n, okay?

00:22:38 Speaker 1
The question is, what is the relationship between Cauchy sequence and convergent sequence? Here it's a classical example in which if you consider the set Q, which is the set of rational numbers, with the absolute value as a norm, and you consider this sequence, so the sequence defined by u n equal to 1 over 0 factorial plus 1 over 1 factorial and so on and so forth, up to n,

00:23:17 Speaker 1
is a Cauchy sequence in Q, but it converges to the number e, which is not in Q, because e is not a rational number, okay? So, in this case, we have a sequence which is Cauchy, but it's not a rational number.

00:23:54 Speaker 1
It's not convergent, at least in Q, okay? So, here comes into play the concept of completeness. A space is said to be complete if every Cauchy sequence is also convergent. Why this is important? Because completeness is a guarantee that when we are analyzing our algorithm,

00:24:32 Speaker 1
or when we are devising our theorem, we are always within the space. So, each sequence that we... ... ...create, for example, for proving something, is a sequence... With a limit that is still in the same space, and this is important that if we want to prove that something about the convergence of some algorithm so once you have defined the concept of completeness, we can move from.

00:25:25 Speaker 1
So, up to now, we have the vector space in a product space, then normed space. Actually, when you have an inner space, you have also normed space because an inner space induces a norm. And then we can move to further definitions. The first one is the concept of a balanced space. A Banach space is essentially a complete normed vector space, so be careful, a Banach space is a space in which we have defined a norm, not necessarily an inner product, whereas an Hilbert space is a complete inner product space.

00:26:18 Speaker 1
So, it is clear that all Hilbert spaces are also Banach spaces, because an inner product induces a norm. The other way around, it's not true. And essentially, the importance of Hilbert and Banach spaces is exactly what I was mentioning before, the fact that when we deal with these kind of spaces, we are sure that they are closed under limits of sequences.

00:27:05 Speaker 1
So the limit of sequences is still an element, the limit of a sequence is still an element of the space. Here we have some examples of Banach spaces. So the space Rn with the norm, the p-norm, or any p.

00:27:37 Speaker 1
greater than or equal to 1, the same for the space LP and the space C, 0, with the infinity norm, so space of continuous functions. These are Hilbert spaces, so it's Rn with dot product, so we have essentially the 2 norm, the same for L2 and H1. We will see this space in a moment. So if we want to give a.

00:28:17 Speaker 1
pictorial view of the hierarchy of these type of spaces, we have the vector spaces, normed linear space, Banach space and Hilbert spaces. Now here there is another point. In calculus 1 and 2, you have seen the concept of the Riemann integral,

00:28:52 Speaker 1
which is suitable for many applications, but it has some drawbacks. And in practice, for I would say most of the applications, it's not enough. What are the reasons? OK, first of all, let us start with a brief recall of the Riemann integral. If you have a function, let's say on an even interval a, b.

00:29:29 Speaker 1
we can say that we can define two quantities, a lower sum and the upper sum, that are obtained by partitioning the x-axis into many intervals of equal length, that we can equal, or not necessarily equal, but for simplicity we can imagine that they are equal, so we can call this amplitude delta x, and on each of these intervals we have the function that.

00:30:04 Speaker 1
will, so if this is the function, and here we have an interval, in this interval the function is attaining its minimum and its maximum. So we can define two rectangles, One by using the minimum of the function in that interval, and one by using the maximum. The sum of the areas of all the rectangles built using the minimum is called the lower sum.

00:30:41 Speaker 1
Similarly, the sum of the areas of the rectangles built with the maxima over all the intervals is called the upper sum. If you take the infimum of the lower sum and the supremum of the, sorry, the infimum of the upper sum and the supremum of the lower sum when delta x goes to zero.

00:31:18 Speaker 1
And if you check that these two quantities are equal, then that particular value is called the Riemann integral of the function, and it's the area below the graph of the function. It's simple, intuitive, and it works for many functions. Let us see a classical example.

00:31:51 Speaker 1
where this kind of approach doesn't work. Here is the so-called Dirichlet function. That is defined to be 1 for x belonging to the rational and 0 if x does not belong to the rational number. So here it's a pictorial view of the function. It's clear that in this case, for any partition that you pick, the lower sum is 0 and the upper sum is 1.

00:32:37 Speaker 1
So, in practice, we have that the lower sum and the upper sum, no matter which kind of partition you are using, are the same. They are always different one from the other. So, in this case we say that the function is not Riemann integrable, and this is actually. something that is annoying because this is quite a pathological function, but similar.

00:33:14 Speaker 1
functions can occur in practical applications, and so it's limiting to have an integral which is not able to deal with this kind of situations. Actually. This is not the only reason why this example is a motivating example for trying to devise an integral which is more general.

00:33:56 Speaker 1
But this is not the only reason. There are other reasons that ask the same importance. So first reason is to be able to deal with this kind of strange functions. Then the multidimensional integrations. So essentially, when you are in dimension, for example, if you want to integrate surface and you have the integral of.

00:34:35 Speaker 1
over a domain omega of the function of x and y, we want to have to devise a concept of integral where it is easier to prove the Fubini's theorem that allows essentially to write this integral as the double integral over x and over y of the function.

00:35:07 Speaker 1
Okay, to split the integration. And moreover, the other important thing is that for many applications, we are interested in swapping the integral with the limit. So for this swapping, If you use the Riemann integral, it is tricky in many situations, and it's not always valid with the extension that we are going to introduce, even if from a very intuitive way, you can also perform this operation.

00:36:02 Speaker 1
The generalization that we are going to consider is the so-called Lebesgue integral. What is the idea? Again, this is a very intuitive way of introducing the Lebesgue integral. The formal way requires much more time and we don't have enough time to enter into the details, but on the left we have the classical Riemann integral that we have seen lower sum, upper sum,

00:36:36 Speaker 1
if for data x going to zero they are the same, we say that the function is Riemann integral. And essentially what we have done is to split into many intervals the x-axis. The basic idea of the Lebesgue integral is to revert the process. So instead of splitting the x-axis we are splitting into intervals.

00:37:08 Speaker 1
the y-axis. And then we have, essentially, once, for example, let us consider this interval, we have to consider on the x-axis, we have to compute, essentially, on the x-axis, the measure of these sets. So, once I have this intersection and this intersection, I have to measure, I have to give meaning to the measure, in this case, the length of this set, and to this set as well.

00:37:46 Speaker 1
So, for example, in this case, I have, for a given slice, I have this joint set to measure. One is here, and one is here. Okay? And... Why? And then, essentially, what you have to do is to compute, if you want, these two intersections and you come up with the corresponding interval on the x-axis.

00:38:30 Speaker 1
We take the two intersections and we measure the corresponding delta x. Exactly. But why this is, where is the important point? The important point is that you are moving from a situation where, as we have seen here, we are picking a small m or capital M, which are the minimum of the maximum of the function in a given interval, times delta x,

00:39:05 Speaker 1
to a situation where. We have the integral of the function is given by a sum of this alpha i times the measure of s i. And what is s i? It is the set of x such that f of x is the interval alpha i, alpha i plus 1.

00:39:41 Speaker 1
So it's in this interval, okay? As you can see, also from the point of view of writing, here we have written f in d mu. Why? Because here the point is that we have to be able to measure set of points. okay to give a meaning of the measure of the measure of a set of points and here enters into.

00:40:21 Speaker 1
play uh all the chapter of analysis which called the measure theory in which essentially you devise way method for measuring set of points but apart from technicalities there is one thing that is very very important is that every countable set has measures zero uh what is in general a.

00:40:57 Speaker 1
measure a measure uh is a generalization of the natural concept that we have about nature, is zero if the set is the empty set and it has the additivity property. So the measure of the union of some sets A, I disjoint is the sum of the measure.

00:41:28 Speaker 1
Okay, coming back to the previous example, the Dirichlet function, here the set of rationals is a set, a countable set, so it has measure zero. So it means that in the computation of the integral,

00:42:02 Speaker 1
The contribution of the rational's point is zero and so in this case, according to this definition of integrals, we have to compute what? First, the value of the function. In that case, we have just two intervals, one where the function has value one and one where the function has value zero.

00:42:35 Speaker 1
So when the function has value one, the measure is zero. And then when for the rest of the points, we have the function is zero and the measure is whatever. OK, in that case. So in this situation, with the Lebesgue integral, we are able to give a meaning also to the integral of that particular function. And this is just an example. And actually, this concept, the fact that the countable set as zero measure is the key point for being able to deal with that pathological situations.

00:43:28 Speaker 1
So here is the computation. So given the function, you have the integral of that. Function. see q that is one times the measure of q over the integral zero one plus zero times the measure of.

00:43:59 Speaker 1
what is left the remaining part q the irrationals on the interval zero one so you have one times zero plus zero times one which is equal to zero so we have a we have been able to come up with a value of this integral in a meaningful way so here we have just.

00:44:34 Speaker 1
Seeing that with this definition, we are able to compute the integral for this string function, which is the first point. The second point, it is possible to prove that with the Lebesgue integral, proving this equality is easy and very elegant.

00:45:04 Speaker 1
Somehow. And then we have these properties, the monotone convergence theorem. So if f n is a sequence of function that converges monotonically. So it means that f one, f two, f n and so on and so forth are increasing or decreasing depending on that. The. of the values, then you can say that the limit for n going to infinity of f is the integral of the limit, okay?

00:45:49 Speaker 1
And the dominated convergence theorem, so essentially if the generic term of a sequence of functions is smaller than g and fn converges to f, this means almost everywhere, meaning that it converges up to a set of zero measures.

00:46:23 Speaker 1
Then we have again the same property. In particular, these two properties are important for what we are going to see about the neural networks. Once you have defined the Lebesgue integral, you can define the so-called LP spaces.

00:47:03 Speaker 1
In particular, the generic LP space is the set of functions that are defined on a set of Mi or Omega with values in R, such that the LP norm of the function is finite. We are that norm is even by these expression is if P is a finite number or for L infinity is the supremum of the absolute value.

00:47:44 Speaker 1
So L1 is what is so if you pick the L1 P equal to one here, we are the functions which are absolutely integrable. L2 square integrable functions finite energy in the sense that if here you have the square,

00:48:15 Speaker 1
that norm is a finite number. This form is related, for example, to the energy infinite element. If you, For example, consider the case where you have an elastic cord fixed at the extrema and you define the displacement u as the vertical displacement. The norm of u is somehow related, the norm, the two norm, L2 norm of u is related to the internal energy, elastic internal energy of the cord.

00:48:54 Speaker 1
L infinity are bounded functions because if the supremum is a finite value it means that they are bounded. It is possible to prove that for p ranging from 1 to infinity, The spaces L, P are Banach spaces. Actually, for P equal to two, they are also Hilbert spaces,

00:49:31 Speaker 1
so you have not only a norm, but also an inner product, which is given by, this expression with the corresponding norm. So why this is important? Because in particular L2, since in L2 we have both the inner product and the norm, we can define all.

00:50:09 Speaker 1
the classical geometric properties that we are used to, such as orthogonality, projections, and so on. um which is which are very important when you want to check convergence of, some or approximation then if omega is a domain with planet measure then we have this inclusion.

00:50:43 Speaker 1
uh so here is the picture and so l infinity is contained in l2 which is contained in l1 and actually the spaces the space of continuous function is something that is uh including, a part of all these spaces okay.

00:51:17 Speaker 1
Then we have to, exactly as we have discussed for the integral, we have also to devise a strategy for dealing with derivatives of functions, particular functions. Actually, it's not that true that we have to consider particular functions.

00:51:51 Speaker 1
So let me just give you an example. Suppose that you have the previous elastic cord, which is fixed at both ends, so it's in zero L. And you apply here, in the midpoint, a vertical force. Directly, the... Vertically and in the negative direction, which is qualitatively the configuration that you are expecting for the equilibrium of this chord.

00:52:31 Speaker 1
Try to make an ideal experiment. You have an elastic chord. You push your finger in the middle, which is the configuration that you have at the end. No, no, with the force. So you push the elastic chord. What is the configuration that you get? So, for example, if I place an heavy ball here and this is an elastic chord, what is the configuration that you have?

00:53:04 Speaker 1
Suppose that this ball is of measure zero, but it has max. OK. it's a triangle it's something like this okay do you know what is the supposing that all the constants elastic properties and so on and so forth are.

00:53:36 Speaker 1
equal to one what is the equation that gives you the value of the displacement the vertical displacement in when you know the forcing term the elasticity equation yeah.

00:54:08 Speaker 1
You have to use the equilibrium equation and then a constitutive equation. This is an elastic cord, so the constitutive equation will be something elastic like the hook law, or generalization for an elastic material, and then you have the equilibrium. And the same through, similar to the heat equation, you have the equilibrium plus the.

00:54:41 Speaker 1
Fourier law for the heat. At the end, what you have is minus second derivative of U equal to L. In this case, R for the temperature minus the second derivative of T equal to T. So here we want it.

00:55:14 Speaker 1
So this is the equation. Do you see a problem with this equation and this force? What is the problem that you can see? Here you have the equation. This is the equation that we have derived simply by considering the equilibrium relation, and the constitutive equation for the core, for the elastic core, and F is the forcing term.

00:55:47 Speaker 1
Now I'm telling you that the force is given by this force concentrated in a single point, for which intuitively we know that the solution has this shape. What is the problem that you can see? I'm not moving unless somebody will come up with the.

00:56:21 Speaker 1
answer because... So this is u of x, okay? We have decided that... So this is the configuration of u of x. What is the problem? You can differentiate once and this function will give you... Suppose.

00:56:59 Speaker 1
that this slope is minus one and this is one. You get something like minus one and one. Then, if you want to compute the second derivative, you are not able, because this is a discontinuous function, okay? So, even for this simple situation, which is a physical situation, you have a problem, because apparently there is something that is not coherent between what you have derived from the physical point of view.

00:57:34 Speaker 1
and the solution that you want to achieve, okay? This is the reason why we have to not only generalize the concept of integrals, but also generalize the concept of derivatives. And, in particular, here there is exactly the example that I was mentioning. So the idea is to try to come up with something that is able to give a meaning to the derivative also for this function in every point of the interval.

00:58:22 Speaker 1
What is the idea? The idea of the weak derivative is simple, conceptually, and it amounts to introduce. The function that, uh, uh, here it's called, uh, he, uh, which is, uh, compactly supported function and, uh, uh, to apply essentially the, uh, integration by parts.

00:59:05 Speaker 1
So, um, the first derivative, uh, the weak first derivative of a function. Uh, so if you have, uh, you prime, uh, and, uh, uh, you prime, suppose it's, uh, the function that we have seen before with a discontinuity and you want to give a meaning to the derivative of this function. What you can do is to pick a function T, which is a very, very regular, uh, in principle here it's written C1, but you can pick also more regular function C infinity.

00:59:45 Speaker 1
For example, you. Multiply the function by the original function u prime, and then since this is compactly supported, it means that it is zero at the ends of the interval, you can apply the integration by parts. And essentially, the derivative that was applied to the original function u goes to the regular function p, which is at least c1.

01:00:26 Speaker 1
So it's u p prime in the x with a minus. This is the integration by parts and the definition of the weak derivative is exactly the same. So we say that G is the weak derivative of U if we have this equality,

01:01:04 Speaker 1
which is nothing more than an application of the integration by parts. And the function phi here is a function in this space, which you can consider essentially of functions which are smooth. These functions are C-infinity, so very very regular, and they are compactly supported, so they are zero at the extreme of the interval so that you don't have the boundary term in the integration by part formula.

01:01:54 Speaker 1
By exploiting this idea, you can define the concept of weak derivative, where obviously this integral has to be intended in the Lebesgue sense. So, let us take this function, which is.

01:02:27 Speaker 1
the function u 3 minus the absolute value of x, which is the function that we have seen before here. The weak derivative is given by this function. And this is true even if in the point 0, in x equal to 0, the function u is not differentiable in the classical sense because you have two values.

01:03:05 Speaker 1
What are the relationships between weak and classical derivatives? You have the same. properties, so linearity, product rule, chain rule, and moreover, if a function is differentiable in the classical sense, then the weak derivative coincides with the classical derivative.

01:03:36 Speaker 1
Now, having introduced the weak derivatives, we can introduce also the so-called Sobolev spaces, which are defined in this way. So, WKP of omega is the set of functions which are in LP, so this index is related to somehow the regularity of the function, sorry, the space, the index of the space where the function u lies.

01:04:19 Speaker 1
And then K is related to the regularity of the function, so it means that the function and all its derivatives, we can say the function and all its derivatives up toward the K are in LP. Okay, in words, we can say that the space WKP is this.

01:04:50 Speaker 1
Okay, we are the norm that we are using. for this space is the sum of all derivatives for alpha ranging from 0 up to k. So we are, like in C1, the norm is the value of the function plus the value of the derivatives,

01:05:25 Speaker 1
the integral of the values of the function plus the integral of the value of the derivatives. And there is a particular instance of the Sobolev space, which is the case where p is equal to 2. So if you consider this Sobolev space for p equal to 2, we have the so-called HP or HK spaces, which are very important for practical applications.

01:05:57 Speaker 1
For k equal to 1, for example, you have the set of functions which are in L2, and the first derivative, so the gradient, is in L2 as well. The inner product is as usual the product of the function and the product of the gradients, and the norm is just the square root of essentially this quantity that can be written as the norm of u in L2 plus the norm of the gradient of u in L2 squared.

01:06:44 Speaker 1
Essentially you have that split this integral into two parts. This is the norm of u in L2, this is the norm when u is equal to b, the norm of the gradient of u in L2. Obviously for k larger than one you have h2, h3 and so on which are important spaces as well but.

01:07:14 Speaker 1
in practical application I would say that h1 and h2 are the most important spaces that we can consider. The generic space WKP is a Banach space, for p ranging from 1 to infinity excluded, and the space HA is an Hilbert space. Essentially, by introducing the weak derivatives,

01:07:52 Speaker 1
we are able to consider and to deal with the non-smooth functions, so functions which are at kinks. If you have taken any course, is, I guess, that you have seen these kind of spaces, in particular, HK spaces,

01:08:26 Speaker 1
when you have, for example, to prove the existence of solutions of a particular equation, you are working with HK spaces. And, in particular, H1, the functions in H1 are continuous for D equal to 1. So, if we are in one dimension, continuous functions, H1 is equivalent to C, space of.

01:09:03 Speaker 1
continuous functions. For D equal to 2, The function in each one might have isolated discontinuity, so they can be discontinuous in some points. Some points means a set of zero measure, because a set of points, even if these number of points are infinite, is a zero measure set.

01:09:33 Speaker 1
So it's still in each one, and we might have isolated discontinuities. In D equal to 3, what happens is that you can have functions which are discontinuous along curves, so lines or curves in space, okay? So, the equality between continuity, continuous function, and H1 is valid only in the 1D case.

01:10:20 Speaker 1
Then we have to introduce another concept, which is nothing more than a generalization of the concept of... Actually, a matrix is a particularization of a concept of a linear operator. But here we are moving the other way around, from the particular to the general. So, a linear operator essentially is...

01:10:53 Speaker 1
An object that takes an element in a vector space x and it gives an element of a vector space y that satisfies this property. So where alpha and beta are scalars, so real numbers, u, v are elements of the space x, and you have this linearity property. We say that the linear operator is bounded if the norm of t applies to u.

01:11:32 Speaker 1
Essentially, here you have the linear operator, we apply the linear operator to u, which is an element of x, you get an element of y, so here the norm is the norm in the vector space y, is bounded by a constant times the norm of the argument, which is in x. Okay, so in this case we say that the linear operator is bounded. We have to give a meaning for here, since x is a vector space and u is an element of the vector space, the definition of the norm here is one of the norms that we have seen before.

01:12:26 Speaker 1
This is another vector space. So we have a definition of the norm that we have seen before. We can define also the norm of the operator itself, which is, if you remember, this expression should remind you of something that we have already seen for matrices. When we have seen the P norm of a matrix. Here we are defining the norm of the linear operator, which is a generalization of the concept of norm of a matrix. A matrix is a particular instance of a linear operator.

01:13:11 Speaker 1
So, in this case, we have that the norm of a linear operator is the supremum for u different from zero of t u, the norm of t u in y, over the norm of u in x. This is the reason why we are requiring that u should be different from the null vector.

01:13:42 Speaker 1
Essentially, why bounded operators are important, because you have a guarantee that when you apply the operator t, you don't have a blow-up, so the objects that you are getting are somehow bounded, okay? And this is exactly what happens with matrices when the matrix has a finite number.

01:14:16 Speaker 1
What is a linear functional? So we have defined a linear operator, which is nothing more than a generalization of the matrix where x and y are not Rn, but generic vector spaces. And a linear functional is an object that starts from space x and gives you a number.

01:14:49 Speaker 1
And by introducing the linear functional, we can define also the so-called dual space x prime, which is the set of all bounded linear functional on x. So x prime is the set of L which are linear functional that are bounded. So it means that we have essentially the same definition that we have seen before.

01:15:21 Speaker 1
The norm of a linear functional in the dual space is the supremum. Remember that this is a number according to the definition that we have seen there. So here we don't have a norm, but it's just the absolute value over the norm of u in x. So the dual space is a bank space. And, intuitively, you can imagine the dual space as the space that represents all the possible measurements that we can do on the space X,

01:16:03 Speaker 1
because L is the set of all linear bounded functional defined on the space. Then, here is another important point, which is the connection between linear functionals and scalar products.

01:16:35 Speaker 1
As you can see, linear functional are defined as quite abstract objects, so they are, it's a map between, it's a linear operator between X and R, but the representation theorem allows to create a sort of connection between linear functionals and scalar products.

01:17:06 Speaker 1
So, if you have an Hilbert space and L is a bounded linear functional at Hilbert space, then you have a unique element U in the Hilbert space, such that the application of the linear functional on V is equal to the scalar product between V and U.

01:17:37 Speaker 1
And moreover, the norm of L in the dual space is equal to the norm of U in the Hilbert space. So in other terms, we can say that in a Hilbert space, a linear functional is essentially an inner product. So we are moving from a very abstract object to something which is more manageable. And here we have some examples.

01:18:14 Speaker 1
Let us suppose that we consider this situation in Rn. So you pick a... vector x, any vector x in n, and you fix a vector y. So y can be interpreted in our context as the weak vector, for example, which is fixed. You fit f with x, and the result is x dot y.

01:18:49 Speaker 1
Okay? So it's a number. It's a scalar product between x and y. So, for example, if you have this linear functional, so the sum of all the components of the input vector x, this can be represented as a scalar product between x and the vector.

01:19:22 Speaker 1
of all ones. Okay, so according to the definition that we have seen before, it is possible to find a unique u, such that there is d equality. So in this case the unique u is this one. Okay, other situation consider the functional l of f which is the integral between 0 and 1 of f of x in.

01:19:53 Speaker 1
dx. We can always find a unique u in l2. 0, 1, such that this integral is equal to this scalar product between 0 and 1 of f of x times u of x. What is u is a function which is 1 in 0, 1 half, and 0 in the remaining part of the interval.

01:20:24 Speaker 1
Finally, and this is going to introduce what we will see about the Svejnko theorem, the question that we want to answer is, is it possible.

01:21:00 Speaker 1
to approximate any function with a simpler function, like ReLU, hyperbolic tangents, or other functions. In neural networks, we are actually approximating any function. The relationship between the input and the output of a neural network is, at the end of the day, a function.

01:21:32 Speaker 1
So we want to approximate this relationship between input and output. And in that case, we use the simpler functions, the composition of the functions, the activation functions, that are present in the different hidden layers. And that is the purpose of the network. If you approach the problem in a classical sense, you have, for example, the approximation theory, where you can use the Lagrange polynomials or the Fourier basis or whatever for approximating a function.

01:22:16 Speaker 1
Or you can use any kind of compression algorithm that we have mentioned when we have dealt with the PCA, essentially, or SBD. So you can use low-rank structures hidden in the data for approximating your data set. The key theorem in this context.

01:22:49 Speaker 1
It is, for example, when you can see the polynomials. is that any continuous function given in a closed interval can be uniformly approximated by a polynomial. So, for f in C , given a positive tolerance, you are always able to find the polynomial P, such that this norm is smaller than epsilon.

01:23:19 Speaker 1
And this relationship. is at the base of, or if you want, it's an application of another important concept, which is the concept of density. And this term will come back in the next lecture.

01:23:52 Speaker 1
when we will consider the Spanko theorem. The theorem, what does it mean that set S is dense in X? So X is a normed space, and S is a subset of X. What does it mean that S is dense in X? It means that if you pick any U which is in X, in the larger space,

01:24:29 Speaker 1
and, a tolerance epsilon positive. you can always find an element of s which is such that the norm, of the difference between u and s is smaller than epsilon so in other terms you can say that, if you pick any element in x you are always able to find an element in the subset.

01:25:06 Speaker 1
s which is as close as you want to the original point u so in other terms in words we can say that a family of functions is dense in space if independently of the target point so u, And no matter how small epsilon is, you can always find s that is close, as you like, to the original point u, okay?

01:25:50 Speaker 1
Examples. Polynomials are dense in the interval a, b, which is what? Exactly exactly this result. This result is stating that the space of polynomials is dense in c of a, b. Then we have that the trigonometric polynomials are dense in the space L2 of 0.

01:26:23 Speaker 1
to pi, which is the result that allows to devise the Fourier series. And the continuous functions are dense in LP, okay? What we will see as a result, so the Sibenko Theorem essentially states that, if you have target space x, which is the target space where you have.

01:27:00 Speaker 1
the ideal function that you want to represent with your neural network, the representation of that function by using the combination of. activation function multiplied by suitable coefficient is dense in X.

01:27:36 Speaker 1
So these will be the results that we will prove. So that the set of functions of this type, and we will make some requirements on the shape of sigma of the activation function, is dense in the original space where the function that we want to approximate lies. This is the reason why density is very, very important.

01:28:08 Speaker 1
And there is another point that I want to mention here. I have not explicitly denoted the norm, but obviously it depends on the space in which you are considering U and S. In particular, that norm will be the norm in the space X, because S is also a subset of X. So it has exactly the same norm. So to sum up, the important concepts that we will need for proving the Svienko theorem are density, compactness,

01:28:53 Speaker 1
Uh, and, uh, uh, convergence, uh, and then, uh, uh, we will also, uh, use some, uh, definition about, uh, Hilbert spaces. Uh, you've got spaces me, uh, for, uh. Proving some properties about, uh, uh, these particular functions. Uh, okay, so we can stop here for.

01:29:23 Speaker 1
The radical part, and so maybe we can have a 10 minutes break and we will start with the. The project, uh.
