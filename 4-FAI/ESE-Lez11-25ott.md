00:00:06 Speaker 1
Okay, so one little remark about last time. So we have seen together exercise 1.2. In the afternoon, I showed exercise 2.2. Basically, they were very similar. Like I have the text and I write the search problem starting from just the textual version of the problem. So if you have tried to do the 2.2 or I suggest you do that, and then there was like a little comment more.

00:00:40 Speaker 1
on a heuristic that was suggested. So just try to double check it and you can see the recording to see the solution that I showed or we can just afterwards or when you do the exercise and you have some doubts, we can look at it together. So I switched them between the two lectures so that you have both solutions. okay then let's start with today's topic so in the previous two lectures we talked about, basically search problem we define them we try to solve them the first time we just said okay let's.

00:01:15 Speaker 1
go with the definition with the everything that it's there and then in the second lecture we said no we want to be somehow have some suggestions because we have them so let's try to exploit them and we use the heuristics so informed search strategies do you hear an echo or just.

00:01:46 Speaker 1
let's try like this okay, Okay, so today we are saying, but probably I can do something more because my definition of a search problem was very, very general. So, okay, we didn't be informed versions. I could, I could like have information. So try to be faster to find a faster solution, or at least try to find an optimal one and so on. But we were still very general. So basically, the search problem was our topples set of states, initial state set of actions, actions function.

00:02:36 Speaker 1
So which actions I can play in that state without function. So which is the result of playing a certain action in a certain state will test. To say OK this state is a goal or not and we discussed that that is a function and then cost function step cost function that tells us OK to play detection in the state as this cost and so today we will talk about constraints satisfaction problem so CSP and here we will have let's say two main ingredients so a factorized state representation.

00:03:55 Speaker 1
And some constraints, so we will have like a specialized sort of search problem where the state must have a certain representation and let's say the actions function. So the possible things that I can do, and it's not just the actual function, because it's not what can I do, like, depending on how you define the state, but it's not like how what are the possibilities that I have for a certain we would call it variable.

00:04:36 Speaker 1
But it's what are the combinations that are allowed. So basically we are. Constraining more the problem we are stricting the number of problems that we can formalize as a constraint satisfaction problem and the hope is that by doing this we have a faster way to find a solution so let's be a little bit more formal so a CSP can be modeled with three elements that are the initial state x that let's.

00:05:27 Speaker 1
So X D and C so X is our set of variables. D is our set of domains and C is our set of constraints so basically this is the state right because we are saying my state is factorized into a set of variables and so and these variables have these domains that are allowed so the I will be the domain of variable I and okay so this is like the map.

00:06:29 Speaker 1
Being of a state we can think about it like this so basically variable value pairs and. The constraints will be indeed their interactions so basically somehow with these formalization I may recall depending on how I define what is my action here but I may recall that I can have a correspondence one to one with a graph because I have certain values that are allowed and other values that are not.

00:07:09 Speaker 1
So in this case the intuition is that what can I do? I may say OK I have an initial state that is all variables are not assigned they don't have a value and I may say OK assignment one is my action a one so here I may say OK X one equal to one X two equal to four and so on.

00:08:02 Speaker 1
So I may assign all the variables and trying all the possible combinations if one of them is is fine with respect to all the constraints then I have a solution but of course this is combinatorial so we we don't like it another option maybe I have the initial state and I say OK at any level I assign one variable.

00:08:33 Speaker 1
So, for example, here, X1 equal to one X1 equal to two and so on for all the values that are in the domain of X1. And then I may say, OK, here I do, I don't know that first. So here X2 is equal to four and then X3 is equal to five and so on. So we can do use everything from bread first to a star as you've seen it and these maps nicely.

00:09:08 Speaker 1
The only thing that I have to care about is that what is a solution state. And here you see that the goal function is better when it is a function because I don't want to enumerate all possible combinations that are fine with respect to my constraints. So the goal check will be OK. Is my. Final note. Where I have assigned everything so every time I have to pay a full path and my final goal test will be is my set of values fine with respect to all my constraints if that's true the goal function will return true otherwise we return false so in this way we could get back to everything we have already seen but of course since we have the constraints we hope to do something a little better.

00:10:05 Speaker 1
So again this is fine but we don't really like it by itself and so let's try to be a little bit more precise with the definition of constraints to try to exploit them in for an algorithm so let me call a constraint. So a certain constraint OK and I may define CJ with two elements SJ and RJ so for the next five minutes let me just define a constraint that may involve any set of variables then in the exercises we will just look at binary constraints.

00:11:01 Speaker 1
But in principle you may have a constraint that involves anything so here we have SJ which is a subset of the variables and this is called the scope of the constraint. And basically this is telling me OK the variables the variables that are involved in my constraint are these variables so the per constraint may have X one X five X nine and they will be my indices that I have here just K1 KJ and of course the let's say cardinality of this set so KJ is called arity of the constraint.

00:12:01 Speaker 1
So just the number of variables involved and the second element is RJ that we call relation of the constraint and this is a subset.

00:12:32 Speaker 1
Okay of the set of domains so basically it's telling me okay the subdomains that are allowed by this constraint are the k one for variable x k one up to the k j for variable x j so what does it mean is that if I have x one x five x nine and I know that they are all between.

00:13:04 Speaker 1
One and ten okay so these are the one the five and the nine I may end up with a relation that is x one belongs to three five x two belongs to one two three or let's say more properly. So, x1, x5, x9 belong to 2, 3, 5, I don't know, I'm just putting random numbers, 9, 5, 1, 5.

00:13:53 Speaker 1
So, this is a subset of the Cartesian product, because I have one element for each of the variables, and I'm saying which are the combinations of values allowed. So, I'm not restricting each individual domain, but I'm saying, okay, these are the allowed combinations, because I have the Cartesian product. And this is basically the same that you see when you have a constraint that you may have already seen during the lecture. So a constraint between X one and X two, which is one a to be OK is just a generalization with the set of any possible number of variables.

00:14:55 Speaker 1
OK, questions. That's perfect. And I also want to introduce what is an assignment. So basically it's a set of couples variable value pair where the variable belongs to the set of variables and the value belongs to that specific domain.

00:15:42 Speaker 1
Now I will comment on this so an assignment is what I showed before so is like okay I have a certain set of variables that may be all the variables or just a few of them.

00:16:14 Speaker 1
And, For the variables that I am considering, I'm going to have a certain value. So an assignment may be the set X one three X two five X four one. OK, so an assignment is just OK. I tell you one variable as a value, another variable as another value and so on.

00:16:45 Speaker 1
So I may do that for all the variables or I may have a partial assignment like this because I'm not giving a value for all the variables. If I also include X three five and we suppose that we have four variables, this becomes a complete assignment because I have assigned all the variables that I have in my set of variables.

00:17:15 Speaker 1
And of course, depending on the constraints, these may be a consistent assignment because I may have all valid values. So I may have something that assigns a value to each variable and the value is OK with respect to all the constraints. Or it may be inconsistent because I may have something that, for example, if I have a constraint x2 different from x3, this is inconsistent because I have assigned the same value.

00:17:53 Speaker 1
So if I am introducing all this stuff because a complete, what is a solution of a constraint satisfaction problem is a complete, consistent assignment.

00:18:28 Speaker 1
Okay, so this is my goal. I want to find a complete, consistent assignment that intuitively is very easy. So, a value of the domain associated to a variable for all the variables such that the constraints are satisfied. Here is a little bit more formal. Okay, so let's go straight to a solution. And today we will see another kind of algorithm.

00:19:01 Speaker 1
So, so far we have just seen search problems, search algorithms. So, okay, I search for a solution with my tree. Now we will start with an inference approach where we exploit what we call constraint propagation. And what does it mean well it means that if I have a constraint I may have immediately some values that are not allowed so I may say okay the domain of my variable is large but if my constraint is like between a constraint between x1 and x2 is 1 a 2 a 3 a.

00:19:57 Speaker 1
Then of course x2 may only have value a so okay if the domain of x2 is r but let's say all the letters from a to z but it's only a value is a so I can. Remove a lot of elements from the valid ones in the domain of x two and so these may propagate to other constraints because in other if in other constraints I have some other letters assigned to x two that are involved they are all not valid combinations and so if I propagate these I may end up with a very restricted domain with respect to what we started and in particular.

00:20:57 Speaker 1
Doing this we will exploit our consistency and we say that excite is consistent. With respect to XJ if and only if all let's say say it like this and then we formalize it a bit better all values that are in the domain of I satisfy all binary constraints between I or let's say XI and XJ so I'm saying that what I was showing before basically.

00:22:07 Speaker 1
So XI is art consistent if in my cost binary constraint I always find the. A value of xj that is okay with any element of the domain of xi and that's exactly what we were showing before so if x2 has domains a b down to z and the constraint between x1 and x2 let's call it x1 is a1 a2 a3 of course x1 is not r consistent with respect to x2.

00:22:54 Speaker 1
Because it's not true that for all its elements I find a value that satisfies the constraint so basically it's suggesting me okay you can remove many elements from the domain of xi because they do not respect they do not they are not good for that constraint. And so, basically here, when this becomes our consistent, when I remove all the other domains and I just remain with A. In this case, it's our consistence because for any value of the domain of X1,

00:23:28 Speaker 1
which is just A, I have at least one element of the constraint that is fine with that value. And no other values are okay in this case. Okay, if we want to be a little bit more formal, we may say that for any element of the I, so it's our consistent if and only if, for any element of the I, there exists an element of the J.

00:23:59 Speaker 1
such that the pair the I, the J belongs, let's say, to the constraint between XI and XJ. we may say, belongs to the relation between this binary constraint. So it's just the same thing with less words.

00:24:31 Speaker 1
Okay, and exploiting this, let me write it. Okay, it's fine. Here, we have our first algorithm, AC3. So 3 is just version 3, and AC is indeed our consistency. So I will try to write it down in a sort of a pseudocode.

00:25:05 Speaker 1
So let's start by including in our... In our queue, which is basically a set of constraints, we include all the possible constraints that we have. So here, we are saying, OK, I want to do inference. So I want to go through the constraints, and try to reduce the domains of each of them.

00:25:41 Speaker 1
as much as possible. This works for binary constraints, because also the definition of our consistency that we have done is on binary constraints. And so we will try to exploit each of the constraints. How? So while queue is not empty, what do we do? We take a constraint.

00:26:16 Speaker 1
Thank you. so we just take from this set one constraint the first one and of course this is a binary constraint and involves two variables x1 and x2 yes yes okay here we apply arc consistency on.

00:26:49 Speaker 1
xi xj okay so we check is this is xi are consistent with respect to xj, And this will be more clear in the exercise in a moment. But in general, of course, if I have a binary constraint, it's not directional. So we will just check for both ways. So is xi arc consistent with respect to xj?

00:27:22 Speaker 1
And then in the next step here, you will have the other way around. So is xj arc consistent with respect to xi? OK. OK, so what and when I say apply arc consistency means, make these arc consistent. So if it's not, then I will reduce the domain of xi, such that for each of its elements, I have a correspondence in the constraint.

00:27:53 Speaker 1
OK, so what may happen? It may happen that di, because I am reducing the domain of xi, Is empty and in this case we may say failure so there is no solution because I don't have any value for x i that satisfies this constraint so this constraint cannot be satisfied okay I don't have a solution otherwise it's not otherwise is if I changed something so let's write it like this so.

00:28:50 Speaker 1
It may also happen that the eye does not change because all the values of the domain are already satisfying the binary constraint so in this case I do not I do nothing. But if the eye has changed okay if the eye has changed for any binary constraint that involves X I I need to check again all the like the art consistency so if all the values that were in that constraint are still valid.

00:30:01 Speaker 1
Because what may happen is that, okay, I'm checking between x1 with respect to x2. I see that if I'm okay with this constraint, then x1 needs to remove two values. But then there is the relationship between x3 and x1. And in that case, if now x1 has a smaller domain, then it may be that a constraint that was valid is not valid anymore because I do not have anymore that value of xi that is fine for a value of x3.

00:30:35 Speaker 1
So in practice, it would be much easier than talking about this. So the point is that when I change a domain, I have to check all the other constraints that involve that variable. And of course, I will double check to see if I can remove some elements from the other variable involved in the other constraints. So the direction is this one and sorry, this was J.

00:31:06 Speaker 1
I don't need to double check again these relationship because I just exploited that. And so, of course, I'm finding values that satisfy the values of X, J. So I don't have to double check the other way around. So I will double check the relationship. I need to check again the relationship with the variable that has changed domain and all the others that were not involved in these check that I'm doing.

00:31:35 Speaker 2
OK, let's look at this in practice.

00:32:19 Speaker 1
Okay, so we have three variables, x1, x2, x3, this is exercise 3.1, the first domain is 1, 2, 3, 4, the second one is alpha, beta, gamma, and the third one is a and b, and then we have three binary constraints.

00:32:59 Speaker 1
So the constraint between x1 and x2 is 1 alpha, 2 gamma, 3 alpha, 4 gamma. The constraint between x1 and x3 is 1a, 3a, 4b.

00:33:32 Speaker 1
And the constraint between x2 and x3 is alpha b, beta a, gamma b. Okay? And we apply AC three so we start by initializing Q with all our constraints in both directions so I need to check whether X one is our consistent with respect to X two whether X two is our consistent with respect to us X one and so on so one three three one two three three two.

00:34:41 Speaker 1
Let me leave this open and let's just say that okay so far these are the ones that I need to check okay so the constraints are in both directions and. Usually, they just tell us which are the possible values. Of course, in an exercise, if you are required to write a constraint, and the constraint is like, I don't know, x1 different from x2, then it's fine also.

00:35:11 Speaker 1
if you write that like this and do not enumerate all possible values. But OK, here, the domains are relatively small, and we can just write down all the pairs that are allowed. OK. OK, so let's take the first one. x1 are consistent with respect to x2. And what does it mean?

00:35:42 Speaker 1
It means that I have to look at the domain of x1 and ask myself. Is there a correspondence between any of the values of x1 in its domain and the constraint between x1 and x2? So one may be possible, yes. Two, yes. Three, yes. Four, yes. So I cannot remove anything because all the values may be valid.

00:36:19 Speaker 1
So here we do nothing and we have checked this one. Now, let's go x2 to x1. So the constraint is the same, but now I have to look at the other domain, because here. And you see that things are different because beta never appears.

00:36:53 Speaker 1
So I have no values of X one where beta is a valid combination. So I will remove beta from the two and this is now are consistent because I'll find gamma out there. So here we remove beta and so this is checked.

00:37:30 Speaker 1
But what happens? We have changed the domain. It's not empty. So we are in the last case and there they are telling us, OK. Okay you have to include in the queue all the constraints that involve the domain that you have changed so now we have changed the domain of x two we have to include in the queue all the constraints that go let's say in x two so the variables are two so we have x one two x two and x three two x two but do we need to add anything to the queue in this case no because.

00:38:20 Speaker 1
X one two x two is the complementary of the one that we have just checked and we said that I don't have to check for the other way around and that's because I am checking the value with respect to the value of the other so I don't need to do the opposite it's already in this check so this one has not to be included. The other one is already there so check it two times makes no sense because it's doing the same check two times it will have sense if I change something else after I have done this check because of course I may need to do the check again if something changes in the domains but now I will not okay so we don't have to add anything else we are going to check this one so X one two X three.

00:39:17 Speaker 1
So I need to check the domain of X one with respect to now these binary constraint so one is allowed three and four are allowed two is not okay so. Since we remove the this one we have looked at x one well I'm not right every time I remove something we will need to include and I will be fast also in this case so x three to x one I will never write it down the only new one is going to be x two to x one do I need to include this yes because it's not anymore there so I have already checked for it but now the domain has changed and so I have to check it again.

00:40:06 Speaker 1
Okay so this one must be included everything clear okay now it's just doing the same a few times till we have a solution so x three to x one. I have to check for a and b with respect to this constraint they both appear so I do nothing three to one and now I have two to three so now I check for the values of the two that are already there are still here I'll find gamma they appear so I don't do anything and I check three to two here a and b.

00:41:24 Speaker 1
Yes a and b they appear so I don't know anything. okay no okay sorry i did a mistake uh because now uh one thing that i didn't add it when i was talking is that we may also remove the constraint that are not allowed so that we may end up with.

00:42:00 Speaker 1
better uh sharper domains so uh in the first case okay we did that we didn't do anything in the second case we removed beta so we can already say okay this constraint that involves beta is not allowed so i can avoid to look at it and the same when i remove the, i removed two so this constraint is useless.

00:42:36 Speaker 1
Okay so in this case when I checked between three and two okay here I have a and B in the original constraint a and B were both allowed but now and this is the reason why we have to check multiple times because now beta is not valid anymore so these value will not be allowed and so the only value that appear in this constraint in is now just be and so I can remove a.

00:43:09 Speaker 1
From the domain of the three okay okay now I like it and since I moved a I need to include the x one two x three. that is not there so i will love it okay now we have the new check between x2 and x1.

00:44:02 Speaker 1
okay so domain of x2 is alpha and gamma i removed a so i can also delete these constraints okay. so one and two now uh okay so alpha and gamma are there i cannot do anything.

00:44:42 Speaker 1
And I check for x one x three okay and this is the other point when I can do something because now between one and three I only have the value for so I can remove one I can remove three I end up with this one so remove one and three pay attention here that x two two x one must be added so the algorithm stops.

00:45:26 Speaker 1
When q is not empty but I check for it when I am finished so here it was empty for a moment because because I. Popped this value, it was empty, but I went through the iteration, I included this one, and so I did not stop, because Q is not empty yet. Okay, so now that I removed 1 and 3, I see that these two are not allowed, okay, and we are almost there,

00:45:59 Speaker 1
because now I have to check between X2 and X1, which is this constraint, and I see that alpha is not allowed anymore. So now I have checked X2 to X1, and I removed alpha.

00:46:29 Speaker 1
And so here I should add the x3 to x2 so let's do that okay and now I see that we have to gamma b is allowed so we are finished because now I extract these I do nothing I do not have anything and so Q is empty okay so sorry if I didn't underline before this point that anytime you reduce a domain you can of course not look anymore.

00:47:16 Speaker 1
The constraint that involves that value that you have discarded okay questions. Now let's look at the solution, the solution is telling us I have a new domain for x1 that is 4, a new domain for x2 that is gamma and a new domain for x3 that is b.

00:47:50 Speaker 1
Of course I was lucky because in this specific case I find three singletons, so one specific exact solution and if I check the constraints of course they are respected, so 4 gamma, 4 b and gamma b. Of course I may end up with... to other possibilities so some sets that are not reduced to singletons so in that case a solution may exist or may not but I have not find it and another way maybe that I end up with an empty domain and so in that case I'm sure that a solution does not exist so if the domain notice this point which is I will also show an example now fastly.

00:48:52 Speaker 1
so if the you end up with an empty domain then okay the solution does not exist but if you end up with some values in the domain the solution may still not exist, Because we are just checking for binary constraints. We are not checking whether all of them may be respected simultaneously.

00:49:23 Speaker 1
When you have a singleton, okay, they are. But when you have a set, it may happen that it's not the case. And the classical example is, okay, I take d1, d2, and d3 that are binary. And my constraints are x1 different from x2, x2 different from x3, x1 equal to x3.

00:50:04 Speaker 1
so these of course cannot be possible because i cannot have x1 and x2 and x3 different both from x2 and b equal between themselves but if you check for binary constraints you say okay here what's allowed zero one and one zero here it's again zero one one zero and here is one one zero zero okay.

00:50:40 Speaker 1
so here with the ac3 i do not remove anything because zero and one so the domains are allowed in both cases, So here I'm checking for pairwise constraints binary constraints but of course it may be possible that they are respected so here I end up with the original domains but if I do a search and I look for a solution it does not exist okay so if I if multiple variables are involved then I'm just checking for binary constraints okay if I find a solution then it's a solution if I find that there is no solution then there is not.

00:51:20 Speaker 1
If I end up with a set of possible values I need to do search so look for a solution and see whether it exists or not and this is exactly what we're going to do.

00:52:27 Speaker 1
You mean here? Okay, so here I have one tree, I remove one tree, so I have the relationship between x1 and x3, I remove one tree, so I have to insert everything that goes inside that goes to x1, so it's going to be x2 to x1 and x3 to x1.

00:52:57 Speaker 1
Okay, and these so this one will not be inserted because it's the same but opposite, and we said that we do not do that. And we didn't do for any of them, so the other way around, you never insert it, you only insert something that goes within x1, but between the other variables and x1, not the one that is involved, while this one has been included indeed, otherwise the queue would have been empty directly in this step, while I included it and directly extracted it to do the last step.

00:53:27 Speaker 1
And here again, we include 322, but nothing changes, and so we don't include anything else. Okay, other questions, doubts? Okay, so let's look at search. So basically, we said, okay, we do inference, and what if we end up with some domains, we do search.

00:54:05 Speaker 1
And in particular what you is usually done in this case is backtracking search be the s so what are its characteristics well first of all is a depth first search so we do that first because we want to go down and find a solution fastly and you may have I mean we have already talked about the complexity so we can do like I have my.

00:54:49 Speaker 1
Initial state I expand it. Generating all the successors I add them to the frontier so I pay a branching factor I have to go down from steps and so I pay be times them in terms of memory because I just need to store one branch but of course also the other successors what we just commented the let's say without implementing in one example was okay I can do it also in a recursive way so I do not generate successors.

00:55:24 Speaker 1
I've been down to the frontier but I just look for an entire branch then I go back and I say okay generate from these the next one. And so in this case, we do not pay any more the branching factor, but we just pay the fact that we have to store one path, and then go back, remove that, generate another, and so on. So in this way, it's called backtracking, because I generate successors, I end up here, I say, okay, this is not a solution, so you that were one before, generate me another successor of you,

00:55:58 Speaker 1
and of course, this can be removed from the memory, and so I only store one entire path in the worst case. So this will be just the only difference by looking at, so let's say, the fs recursive. And of course, the space complexity becomes m.

00:56:31 Speaker 1
And then another characteristic will be OK let's assign one variable and time so in my route I say OK now in the first step we assign X one and the first one is one then two three and then we sign next to and so on so for each step we assign one variable and like yeah the the last thing is indeed that we have the community property.

00:57:24 Speaker 1
So for the algorithm we don't really care whether at the first step we select x one we assign x one or x nine or x seven because it's commutative so the final solution the leaf node that I end up is independent with respect to the order where I did the assignments. So basically we may say that the order of actions has no effect on the solution.

00:58:13 Speaker 1
So we may indeed follow OK now X one the next two the next three and so on because we don't need to we will not have different parts if we assign X one and the next two or X two and the next one so basically if you think about. And this is one example because that underlines that this is a restriction of a search problem and let's see that I am here and I know this is not very let me think about an example OK so let's make it more intuitively.

00:59:10 Speaker 1
I have an action that is like, I don't know, break something so one action is take one thing and put it there another action is, break my thing for example and of course if I break it beforehand I cannot move it afterwards it's a stupid example but just to say that in general in a search problem you may find examples where you start from a point you go to another and then in that case the action that you may.

00:59:44 Speaker 1
have done in the first step is not allowed anymore because we have the actions function so, depending on where we are we can do different actions so if I break something I cannot use it anymore so if in the first step I can both decide use it or break it and in the second step, let's say in both states in principle I may do the same but of course when I break it I cannot use it anymore here is not the case so here anytime okay I assign it but then I go check for x2 so it's not.

01:00:18 Speaker 1
relevant I will not change like what can I can do for the other variables of course okay so with this we also have two additional elements one is forward checking that will help us so let's say after any assignment apply our consistency which means it's not that we are consistent to see that we were applying.

01:01:05 Speaker 1
So AC3 this may be like what we do after AC3 ended up with domains that we have to search for a solution. Here is, okay, in my branch here, I said x1 equal to 1. Now, also if the constraint allowed for other values of x1, now I'm fixing x1 to be equal to 1. So, of course, from now on, valid constraints will only be the ones that include x1 equal to 1 as a value.

01:01:44 Speaker 1
And so, I can, like, prove more the set of constraints. I will just look for x1 equal to 1. But it doesn't mean that I remove them like that because then if here I don't have a solution and I backtrack, I look at the arrow x1 equal to 2. That may still be allowed by the constraints. So, I'm... I'm applying our consistency to the domain of the others once I have done an assignment to immediately check whether it will be possible still to find a solution or with this assignment all the others have no meaning.

01:02:24 Speaker 1
OK and the last point is that we can also apply some heuristics and usually they are minimum remaining value and subsequently least constraining value.

01:03:13 Speaker 1
This one is applied to variable ordering so let's say to do variable ordering I use this to do value assignment or value ordering I use this so what does it mean it means that I'm doing the search in principle then now we see an example where we do all the steps and we see how they behave in practice but the idea is that since I have the commutative property I can just follow x1.

01:03:56 Speaker 1
x2 x3 and so on or try to be a little bit smarter and what may be being a little bit smarter when it may happen that I have. Which variables do I choose the one with minimum remaining values because that one is the one where I may be more like constraint obliged to assign that value so if I have x1 that takes values one two three four and x2 that can only be equal to a then of course I immediately assign x2 equal to a I do not generate all possible branches because these will just be that one.

01:04:37 Speaker 1
So let's use the minimum remaining value while for the least constraining so here is to select between x1 and x2 and I say OK let's go with x2 now suppose that x2 is also big OK so I select the one with the smallest domain and between a and b what is the one that I prefer well if the set of constraint is one a. 2A, 3A, 4B, let's try to assign A firstly, because in that case, it may be possible that I still remain with many possibilities, and so hopefully I find now a solution, while B is constraining much more my search.

01:05:32 Speaker 1
So if I select B, X1 can only be 4, and then maybe X3 has no, again, combinations that are okay. So if I have to choose between variables, I try the ones that have less values that I can assign. If I have to choose between values, I look for the ones that appear more times in my constraints.

01:05:58 Speaker 2
Okay, so yeah, let's go here.

01:06:57 Speaker 1
Okay, so we still have three variables, x1, x2, x3, the first domain is still 1, 2, 3, 4, so this is exercise 3.2. The second domain is a, b, c and the third domain is alpha, beta, gamma. First constraint, 1a, 2b, 3a, 3b, 4b.

01:07:41 Speaker 1
Second constraint, 1beta, 3beta, 4beta, third one, a, gamma, b, beta, b, alpha, c, gamma. Okay and let's see all the elements so let's try starting with just backtracking search so in this case I start with no assignment I decide just to follow let's say the lexicographical order the order of the variables that so I started x1 then I assign x2 and finally assign x3 at the third level.

01:08:36 Speaker 1
So we are doing that first and we are doing it recursively so I just generate the first successor which is x1 equal to one right so I just follow the order now here is I assign x2. The successor is going to be x1 equal to 1 and x2 equal to 8.

01:09:13 Speaker 1
And while we do this, we do not do forward checking. So we do not look at the domains of the others and check whether they allow for a solution. But at least we check whether this is admissible. So 1a, so far, it's fine. If I do not have 1a in the set of the constraint, I just stop here.

01:09:46 Speaker 1
Because at least I check for the constraint. So I'm not adding forward checking and heuristics. But at least I check for the constraint before expanding something useless. Okay, so now I should expand x3, the first value may be alpha, but I see that, for example, one alpha is not allowed, so here, let's write it like this, so I do not generate this node, because one a alpha is not possible.

01:10:25 Speaker 1
Let's try with beta, one beta is okay, so one a is okay, one beta is okay, a beta is not, so also in this case, I do not generate the solution, and if I try with gamma, again, now one gamma is not allowed. So, no solution here, so I tried here, no, I remove this from the memory, and I go with the other one, that is, here, I tried with A, now I try with B.

01:11:07 Speaker 1
So, now I have X1 equal to 1, X2 equal to B, and I try to check, is this allowed, 1B, it's not already, yes, so here I stop, and also C, so X1 equal 1, X2 equal C, so I may also check before generating.

01:11:43 Speaker 1
So, here it was not allowed. And also here, 1C. I try this combination, it's not in the constraint, and so I do not generate this successor, okay? So we are aligned with here that I didn't write them down, okay? So now I have to backtrack once more and try. Here I tried with 1, now I tried with 2. x1 equal to 2.

01:12:15 Speaker 1
Now I have to extract x2. Let's try to assign a. 2a is not there. So here I do not generate anything. Let's try with b. 2b is there. So x1 equal to 2, x2 equal to b. Now I have to assign x3. Let's try with alpha.

01:12:48 Speaker 1
2 alpha, I mean, from these, only beta will be allowed. So here, nothing. Let's try with beta. But 2 beta is not there. So here, I do not generate anything. And with gamma, we said that there is no combination. And so we stop here. So now we backtrack again. We try with c. But 2c is not in the constraint. So we also do not generate this one.

01:13:19 Speaker 1
We go back. No successor anymore. Go back again. And try with 3. Let me do it like this. So x1 equal to 3. Now, I have to extract x2. Let's try with a, OK? 3a is there. So this is fine.

01:13:52 Speaker 1
Okay, and now it's time for x3, alpha, 3 alpha is not allowed, we have said only beta is the one that we really have to check, 3 beta is there, but a beta is not there, so also here no successor, and we gamma, we said no successor, so now we go for b, we say x1 equal 3, x2 equal b, alpha, we have said that it's not possible,

01:14:33 Speaker 1
let's try with beta, so 3b was okay, 3 beta is okay, b beta is okay, so we assign the values. And this is our solution okay notice that these branch with the C has never been generated because we are doing the recursive formulation okay let's try forward checking so what happens here is that I need to decide the variable and again will be x1 first value second value third value and so on but also to take care of the domains.

01:15:36 Speaker 1
Of the others so here the one is one two three four the two is a b c. d3 is alpha, beta, gamma, okay? In the root, I have everything. Now I assign x1. The first thing is 1. So here, x1 equal to 1.

01:16:07 Speaker 1
And what happens is that if x1 is 1, x2 can only be a, right? So you see, I'm not saying the only value allowed is 1a. I'm saying if x1 is equal to 1, so in this branch, the only value allowed is a for x2. So here, the domain of x2 is just a, and the domain of x3 is just beta.

01:16:48 Speaker 1
Okay, so now that I have to generate, so assign x2, of course, I only assign x2 equal to a. Okay, and now I check for the domain here, and the only allowed value is gamma, but I just have beta. So the domain of 3 in this branch is the empty set, and so I stop.

01:17:24 Speaker 1
Okay, so I expanded much less. And I'm stopping just one step before. So I backtrack. Here, you see, I don't have to generate anything else because I only have a in the domain. So I backtrack here, and I generate the values with x1 equal to. Now, if x1 is 2. x2 can only be b, but x3 is the empty set already, so d2 is 3, d3 is empty, I stop, I go here, x1 equal to 3, now if I have 3, I have a and b, and also beta, and you see here that the heuristic would have been better, because I would assign just this one, but now I'm not using that,

01:18:38 Speaker 1
so I will try with x2, generate and assign x2 equal to a. Now combination 3a the domain beta was not combined with a so it's empty d3 is empty I stop and I generate the other one x1 equal 3 x2 equal b okay and now d3 is still beta because I have 3 beta that's okay and b beta that's okay.

01:19:25 Speaker 1
So I generate also that one and I end up with the solution okay let's try. To be faster and include I remove that one yes okay so backtracking search forward checking and minimum remaining value so just the first constraint on the order of variables I start with empty and now minimum remaining value.

01:20:42 Speaker 1
means that I have to assign one between x2 and x3, right? Because x1 has four remaining values, while x2 and x3 have three remaining values. So I have to select between them, it's a tie, and I will just use x2. So first value is a, okay? Now here, if we look at the constraint,

01:21:14 Speaker 1
with a we have one and three, and the three is gamma, this one. Now, minimum remaining value, so I select x3, and here I can only generate gamma.

01:21:49 Speaker 1
But D1 becomes empty because we have said that D1 only allows for beta. Okay, so I backtrack. I can go up to the root and try with B. So X2 equal to B. In this case, D1 is 2, 3, 4. Yes.

01:22:20 Speaker 1
And D3 with B is alpha and beta. Looks much more promising, right? Because we still remain with many values. Now, minimum remaining values. So I go for X3. First accessory is alpha.

01:22:51 Speaker 1
But b alpha means that d1 becomes empty because we have said that d1 only matches with beta.

01:23:00 Speaker 2
So I go back, I try with beta, and this is fine.

01:23:18 Speaker 1
So d1 is still 3, 4 because they allow both. Let's try with 3, and this is fine. Notice that here we have still found the same solution because we are following the order. But also 4 would have been okay.

01:23:48 Speaker 1
Okay, and last one. everything so just when there is a tie I try to like when there is a there are more elements in the domain I try to extract the one that is least constraining so the one that appears in more constraints okay so I start with the root here again the one is one two three four the two is ABC the three is also ABC.

01:24:38 Speaker 1
okay now. Okay, there is a time I select x2 as before, but now what value is the one that is going to be assigned firstly? Well, I need to check. Let's suppose that I select x2 equal to a, okay? What are the cardinalities of the other two domains that remain?

01:25:13 Speaker 1
So, if I select a, well, we did it here, I have two values for d1, one value for d3. So, this is 2 plus 1, okay? If it is equal to b, the cardinality is, let's check. 1, 2, 3, 1, 2, so 5, 3 plus 2, and if x2 is equal to c, we didn't do that, but let's try, so with c, the only value is this one, so it's 0 plus 1, okay, least constraining, so I take the one with the largest cardinality, x2 equal to b.

01:26:17 Speaker 1
Now, the domains are those two, d1 equal to 3, 4, d3 equal alpha beta, okay, we are already in the branch that we like. OK, now, which one do I select X three because as least remaining values, OK, so I will need to select X three, which values alpha or beta?

01:27:02 Speaker 1
Well, if I choose X three equal to alpha, OK, the cardinality of the one is going to be zero because, again, the one X one is only with beta. If I choose X three equal to beta, the cardinality is two because it's this one.

01:27:33 Speaker 1
So I go with the beta. And D1 is now 3, 4. Okay, and now we are happy because X1 is the last one. So, okay, I can extract it. I have 3 and 4, least constraining. Okay, it's the last one, so no constraints anymore. I just select the first one again.

01:28:09 Speaker 1
Okay, and again, we see that both solutions are okay, but we just used one branch. So, of course, this was like a very lucky example where we start from an involved graph and we end up having one individual branch. One little remark is that in the worst case, we didn't improve anything because the worst case is the case where we just have to do that first. So, where all the combinations are allowed in the constraints, so the constraints do not reduce anything. And also...

01:28:41 Speaker 1
Yes, the domains are complete, so I have to do that first in its original version, but of course the mean complexity is much better, so at least for practical uses, as you see, if we combine everything we hope to find a very faster solution. Okay, so I guess we can stop here next time adversarial search.
