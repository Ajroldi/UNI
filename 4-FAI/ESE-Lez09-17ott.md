00:00:10 Speaker 1
Okay, I guess we can start. Good morning. So today we will mostly look at informed search strategies, but I have a couple of remarks about last time and the exercise that anticipated that we will go through before moving to that part. So one minor detail that somebody underlined so we can clarify it better is about the complexity of uniform cost search. So when we went through it, we said that it is big O of branching factor.

00:00:42 Speaker 1
to the power of one plus C star divided by epsilon. And in this case, you can use just, The floor so it's not really needed the seal the very first last time, So to be aligned with the book the floor is fine. And the point is that when you have a certain cost Start you start from zero and basically if your minimum so epsilon was the minimum the minimum step cost that you paid So in one step, let's suppose that I always pay that one in the worst case.

00:01:14 Speaker 1
So in the first step I pay epsilon and so on, Here another epsilon and this final thing. So if we do, Seal of that means that this final thing may be something also smaller than epsilon because I have c star divided by epsilon I have my quantities and then, Something that remains that is smaller than epsilon or like the division, but that's not possible because epsilon is the minimum step cost So either you have perfectly fit of epsilon. So it's just c star divided by epsilon Otherwise this thing must be greater than epsilon. And so basically it's another.

00:01:47 Speaker 1
Let's say, It cannot happen that I have this small thing here, so it's just like this. And so the seal is not necessary. The floor is enough. Because I have all my costs that are epsilon, and epsilon plus something. Not epsilon and then something. So I need the floor is enough. I don't need the seal. I don't know if it's clear. Otherwise, just ask also afterwards. And another thing was that I had some questions about the PDF. So the results that you see there, all the exercises, sometimes have different conventions.

00:02:20 Speaker 1
Like in an informed search that we saw last time, as you see in the exercises, basically most of the time, the frontier, the closed list is checked and not generation and not another expansion. So you may see that the graphs are slightly different and also the search queries. Like the properties of the algorithm. does not change significantly but if you have some questions about what we did so if it's different but still doesn't ring like fine then we can discuss otherwise it may be different for conventions but shouldn't be inconsistent so just if you have any doubts on that we can also.

00:02:50 Speaker 1
discuss that afterwards or if you have specific questions okay so let's move to the exercise that i promised last time so exercise 1.2 so in this case, we have like the first step of our solution so we start from a problem and we try to define it as a search problem and then uh like in the last step we have also applied the informed strategy but the point is how we move from the text to the search problem formulation.

00:03:23 Speaker 1
so in this specific case we have three containers one with capacity equal to 12 liters another one with capacity equal to eight and the last one capacity is equal to three, And we also have a water faucet, so we can take water from outside and put it in the containers. And what can we do? We can fill the containers from the faucet, and this means that I can fill completely the containers.

00:03:59 Speaker 1
So if I choose this action, this becomes full of 12 litres, this other of 8, and the last one of 3 litres. I can empty the containers, so they just become empty, and I can pour into another container. So in this case, if I have this one that is full, this one is empty, I can pour 8 litres. If this already has, I don't know, 3 litres, I will pour 5.

00:04:29 Speaker 1
And if this just has 3 litres, of course I can only pour 3 litres. For the action, we fill it completely, we pour everything we can, or we completely empty the container. Okay, so let's revise the problem formulation as a space problem. So, first element, set of states. Do we have any suggestions? So what am I going to do? I'm going to say I have some liters in the first container, x, some other liters in the second, y, and finally some liters in the third, z.

00:05:05 Speaker 1
Okay? So let's say they belong to the natural numbers. So I have 0 liters, 1 liters, 2 liters, and so on. x is smaller or equal than 12, because it's the actual status of the first container. y is smaller or equal than 8, and z is smaller or equal than 3. Okay, so here I'm trying to formalize it a bit better. It's okay also if we just write down x is the capacity of the first container, y is the capacity of the second, and z is the capacity of the third.

00:05:37 Speaker 1
From these, it doesn't specify the text, what is the initial situation, we say I have the container, so they are empty. And 0 will be 0, 0, 0. And it does not specify even if we have some costs like for water or I don't know, so we just say that we want the shortest path. So the cost for each state action pair will be equal to 1.

00:06:10 Speaker 1
Okay, then set of actions. So what we can do, let's write this. Let's start just with words, and then we try to formalize it better. So first thing that I can do is, so, okay, is fill. So I can fill, like, let's see. right like this, which means I can fill the container with capacity 12, capacity 8, or capacity 3. And then we can do also the opposite. So empty of 12, empty of 8, and empty. And then I have all the.

00:06:48 Speaker 1
pour combinations. So I can pour like from 12 to 8. So I'm writing like this, just write as you as you want. So here means I pour water from the container with capacity 12 liters to the container with capacity 8 liters. I can pour 12 to 3, I can pour 8 to 12, I can pour 8 to 3, 3 to 12, and 3 to 8. Should be all of them. Okay, so this is my set of actions.

00:07:28 Speaker 1
And we can also try to formalize it a bit better. Say that the set of actions is the tuple, like type, i, j, where I would specify what is the type of action that I choose, so field empty or poor, the source container. So for empty and field, I just need the source container. But for poor, I also need the sink container. And so all of these type can be field, empty, or poor.

00:08:08 Speaker 1
i will be 1, 2, or 3. Let's say that I call these container 1, container 2, and container 3. And so these are capacity of container 1, capacity of container 2, capacity of container 3. And i will always be populated here, because I will always do one action with some container, at least one container. While j will be again, one, two, three, or none. Because if I do empty or fill, I don't need the second container. You may also formalize it slightly differently. So instead of saying none, you may say, okay, it's still one, two, three,

00:08:39 Speaker 1
and I do like empty or fill of one, one. So I just use one, for example, and you remove the none. It's fine. Specify what you mean. So if you just don't write none and don't say anything, it seems a little step is missing, okay? And so we have four elements. We have still missing three. And this is just the set of all the actions. But of course I cannot do all the actions in all the states because if I have like an empty container,

00:09:09 Speaker 1
I cannot empty it or pour its water to another one. So the actions function tells me which are the possible actions that I can do in my state s. So intuitively. What I will do, well, I will like empty this if the like empty one container in this state if the container has some water. I will fill it if it is not full and like pour water from one to another depending on.

00:09:42 Speaker 1
how much water I have. So what is the minimum amount of water that I have in the source and how much space remains to fill the second one. So the minimum between these two will be how much water I pour from one to another. So let's try to formalize also this one, not just writing words. So available actions are fill of i. So remember that the action is type i j and i represents one of the three containers. So basically here I have my x, my y, my z and one of these will be i, another one will.

00:10:12 Speaker 1
be j and the other one will just be different from i and j. So basically a will be equal to a certain type. i, j, the state will be equal to x, y, z, and basically i will be one of these three, j will be another of these three, and the type will specify which action I do. So here I have type is field, i is a certain element of the state, and so I can do this, let's say s of i,

00:10:47 Speaker 1
so the i-th element of s is different from the capacity of that element, because in that case, it is not already full, and I can fill it. The same is for empty, so empty if i, if s of i is, greater than zero. So in that case, if I have the container that I choose to empty, must have some water, while the container that I choose to fill must not be full. And finally, pool of i, j,

00:11:22 Speaker 1
so first of all if i is different from j otherwise i'm just using the same container to pull in itself then i need that s of i is greater than zero because the source must have some water otherwise i don't pull anything and that s of j is smaller than its capacity otherwise i don't have room left to pour water in it okay and very similarly we have the result function so result here is sa so these were the actions allowed in state s and this is the next state you know.

00:11:58 Speaker 1
starting from a certain state and performing certain actions so in this case my result is, going to be let's call the next state like s prime so i will have x prime y prime z prime so, let's say s prime of i is equal to the capacity of that container if i'm doing the fill action. and s prime of k will be equal to s of k for any k different from i if the action is fill of i.

00:12:41 Speaker 1
So here I'm saying, just formally, what is really easy. So I'm filling i, the next state of that container, so one of these three, will be its full. The number of water in it is equal to its capacity, while for the other cases it will be just as it was. Okay, complementary, so s prime of i will be zero, and s prime of k will be equal to s of k for any k different from i if the action is, empty of i. So in this case, just an empty i, the other ones will be the same.

00:13:15 Speaker 1
While the last one is I'm pouring from one to another. So let's not consider for a second how much I am pouring. I will just say that S prime of I will be equal to S of I minus a certain quantity, the one that I'm pouring. Then S prime of J will be equal to S of J plus the same quantity because I'm pouring water from here to here. So that one will have the water that it had minus T, and this one, the water that it had plus T. And finally, S prime of K will be equal to S of K.

00:13:49 Speaker 1
So in this case, just the third one, I will not do anything. And this is the action is pour I J. And we just need to define this T. So T, what is the water that I can pour from one to another? The minimum between how much water I have in the first and how much space I have in the second. So the water of the first is S of I. And the space that I have in the second is the capacity of J minus S of J. How much water I already have there.

00:14:24 Speaker 1
I think this is all. And now I have set of states, set of actions, initial state, cost, actions function, result function. I just need the goal test. So here we can just say that goal of S. I didn't say that at the beginning probably. I just want one liter for each of the containers. So goal of S is going to be true if X equals Y equals. Z equals 1. False otherwise.

00:14:59 Speaker 1
Think about this for a second and give me some questions if you have any. Yes? No, no, in the end, all of them must have one liter. So all the three are equal to one. You're good? Okay, so let's go through the solution very, very fastly. And since last time we didn't talk about depth-limited search, we will do that. So initial state is going to be, the text tells you use depth-limited search.

00:15:31 Speaker 1
with depth equal to three. So let's start with depth equal to zero. So I have zero, zero, zero. That is my initial state, so I don't have water. Yes? . Okay, yeah, you're right. Okay, no, so I was wrong. It's, like this, I said everything is equal to one. But yeah, you're right, also the graph that I did was one or the other. So let's say, sorry?

00:16:05 Speaker 1
All one, yeah, I guess so. Yeah, if you pull, yeah, I don't know if you can manage, like, pulling two leads. There's no, because then you have to pull the two there. Yeah, probably all one is impossible. So let's just rewrite it like this. So x or y or z will be equal to one. Yeah, okay, at least the three would be much easier like this. Okay, so what can I do?

00:16:36 Speaker 1
So at the beginning, everything is empty, so I can only fill one of them. So the first action is 12, 0, 0. The second one is 0, 8, 0. And the third one is 0, 0, 3. And here we will do uniform cost. And basically, since it's like a shortest path because cost is always equal to one, is equal to, like, expanding everything.

00:17:11 Speaker 1
So here, if I do uniform cost with L equal to 0, so I just say, you have to check the root is trivial. I just have the initial state. L equals to 1 is this one. So basically, in this case, in the frontier, I have my three values. But then when I try to expand them, I say, no, my limit is 1. So my uniform cost will not be expanded anymore. So it's not like I stop here. While with depth limited, I do depth first. So I say, OK, go down with one branch,

00:17:42 Speaker 1
but down to one. So with depth first, you usually go like this, then like this, then like this. Then you get back into like this. But here, I cut it with depth one. And so I don't do anything else. So I don't do uniform cost here. Otherwise, I will have to expand everything is equal to breadth first. So let's start with this. And here, I could fill y. So 12, 8, 0. I could fill z. So 12, 0, 3, I could empty, so 0, okay, and I can also pour, so good news, we will not need to expand the other two, is 4, 0, because I'm pouring from these to these, but this capacity is 8, so 4, 8, 0, and finally, I can have 9, 0, 3, because the capacity of the turn is 3, and so if I pour from here to there, I will pour 3, and I don't get 9 left.

00:18:52 Speaker 1
Okay, so this is depth L equals to 2. So if I do depth first, but limited at 2, I get here, and I say, okay, now with depth first, I should expand this, but I am at depth 2, so I stop, I get back, I try with this, but I am at depth 2, and so on. So the only thing that you do in this case, if you cut with depth 2, is do the goal check at this depth. If they are not goal, you get back, and you try with this one, because it has depth 1, so you can expand it once more, there you will just do the goal test and get back to them. But here we are lucky because we are doing depth limited search of 3, so we have another round to try to find a goal. Okay, so first one, I can fill, also the third one, 12, 8, 0, 3.

00:19:36 Speaker 1
I can empty one of the two, so 0, 8, 0, or 12, 0, 0. Okay, and then I can pour. What can I pour? Only from one of them to here, because the other two are full. So here I don't have any capacity left. So I can have 9, 8, 3, because I have to put everything, or 12, 5, yes. Okay, now my depth is 3.

00:20:07 Speaker 1
So here I'm doing depth first. This is all my frontier. These ones are in my leap of view, the last node that I added, so the one that I had to check. None of this is a goal state, and I will not expand them because I reached the depth. Notice that since we do not expand them, we do not include them in the close list. And this is correct because if we do, if we do graph search and we have the close list and we include one of these that we have not expanded, we may discard paths and solutions. So this is one crucial point.

00:20:38 Speaker 1
So in depth first, you can do whatever you want, basically, but in depth limited, you do not include these in the close list. You cannot switch between checking if I have reached the depth and include it in the close list. Okay, so now I get back and I have this one. So I can do like this. I can empty the second, sorry, full the second or empty the other two. Okay, and then I can pour from one of the other two to the center one because the other are full. So I can have 4, 3, 4, 12, 3, 0. Okay, then here, I didn't explicitly write the closed list, but let's suppose to do graph search. This is equal to initial state. I do not expand it. And of course, here I did all the checks and none of this is a goal. So I go back. I have that one. I do not expand it. I arrive here.

00:21:34 Speaker 1
And here I have a solution, hopefully. So what can I do here? I can pull this one, 4, 8, 12. I can empty the other two. So 0, 8, 0 or 4, 0, 0. And they are not goals, but I can also pour, yes, from this to that. And in that case, so I can pour from one of them to that one. So in that case, I have 1, 8, 3, and also the other pour.

00:22:06 Speaker 1
that is like 4, 5, 3. And here, I can also pour from this to this, because this is not full. So I also have 12, 0, 0. It's basically the action back to that, yes. Oh, yes.

00:22:36 Speaker 1
And here, I'm down. This is not a goal. I have a goal. So here with depth equal to three, I have the solution. In general, you have the solution if D is smaller or equal than L. So basically, if with breadth first you reach a solution, the shortest path is before your limit. Otherwise, you don't have it because the shortest path to a goal must be longer.

00:23:02 Speaker 2
OK, so let's move to informed search. OK, so so far we have seen a search problem.

00:23:46 Speaker 1
like the one that I just erased. And we said, OK, I also call it the state space problem. And why I do that? Because I can map everything into a graph. I did not do that because the graph would be huge. And I have a search tree to approach the problem. So I have a mapping one-to-one from the formulation to the graph. And then I have the tree that is made of nodes that contain a state but represent a path from the initial state to that one. And in the search tree, I searched for a solution. And we said, OK, to do that, I'm going.

00:24:17 Speaker 1
to need a search algorithm. And basically, we have talked about tree search and graph search, so elimination or not of repeated states. And so here, what was missing was how to order the nodes that I have to explore. And so we have talked about uninformed search strategies. And we have seen examples, completeness, optimality, and complexity of our breadth first, depth.

00:24:50 Speaker 1
first, uniform cost, and depth limited. And we have basically said that this is the optimal one. So if I want optimality, I have to go for uniform cost, because I have to expand sorted by cost. So do my exploration of everything that cost one, then everything that cost two, down to when I found, finally, a goal. So what differs today, all of this will be the same, but we will talk about informed, search strategies. So informed means that I have the problem description, like the problem definition.

00:25:23 Speaker 1
So everything that we did before, because also the path cost, so the cost function, the step cost function was part of the problem formulation. So basically, in the problem definition, you have everything. And this is equal to what we have for uninformed. But we also have some problem, let's call it problem specific knowledge. So a certain heuristic, so I say okay, I have a cost to reach this point, but let's try to find something that suggests me.

00:25:54 Speaker 1
Which may be the cost to go to the goal from here. So like the other part and of course This is a heuristic some knowledge about something related to the problem. That is not exactly what will be my cost otherwise, I have a solution but is something related to that that somehow I have and, So in this case, we will talk about a new element, which we call the evaluation function Which is F that goes from the set of nodes Notice not states but nodes to R. So F of n is telling me from a certain node.

00:26:32 Speaker 1
How I evaluate the path cost so that let me be a bit more specific. So here I have two elements. The first one is g of n, which is the path cost function. And then I have h of n, which is the realistic function, the one I was talking about before.

00:27:04 Speaker 1
And this one, I wrote h of n, but it's basically h of s. And what am I saying? I'm saying that I have a certain initial state. I arrived somewhere in a certain node. And then I have something that leads me to a goal. Until last time, we were just looking at this part, because we were looking at, OK, how much does it cost to start from the initial state and reach the state that I am here now? And this was the path cost, g of n. And of course, this must depend on n because this is.

00:27:36 Speaker 1
the cost to reach that node and we have said that if this is node a and i have another path g prime of n or let's say g of n 2 and this node is still a of course they may differ because i may have different path cost leading to the initial state to the current node so depend because the node represents the path okay and like the new component is the realistic function because now i also want to say how much does this cost or given my knowledge how much can i estimate.

00:28:13 Speaker 1
that may cost from this node to reach the goal and we have already said that, based on the formulation of the problem, I have the action function and the result function. So what I can do here is always the same, independently from the path that I followed, which is this. And this is a restriction because if you think about, I don't know, you have a budget cost, so if you pay five to get here and 10 to get here, and you have a budget of 15, here you have left five, here you have left 10. So it may not always be the case that from there on, you can do always the same thing. In this specific case, my actions function only depends on the state.

00:28:45 Speaker 1
So in this case, the heuristic will be not associated to the path that led me to this state, but just to an estimate of how much costs from this state to reach the goal state. And so here, if I have another path that leads me to a goal, basically the heuristic will be exactly the same, H of A in this case. Okay, and basically the evaluation function will be G plus H, at least in the really interesting algorithm that we will look at.

00:29:15 Speaker 1
While of course, if I have F. that is equal to g so just to the path cost i have if my evaluation is just the path cost that let me here the uniform cost so this is just the ucs and if it is equal to h i have greedy best first that means i don't look at the cost that let me here i just open the most promising node from here on so call this greedy best first okay okay so let's talk a little bit.

00:30:00 Speaker 1
more about realistic functions before showing an example. So, heuristic function, and we have basically two main properties that you have already seen that are needed for the different guarantees that the algorithm are going to have, which are admissibility, so we are going to say that h of n, let's say h of s from now on, is admissible if h of s is smaller or equal than h star of s for n e s.

00:30:49 Speaker 1
So, and of course, this is an estimate of the cost, we know that all costs are greater or equal than zero, so we can also include that one, so my heuristics will always be greater or equal than zero, otherwise it's just a useless heuristic. So the easiest one that basically leads us back to uniform cost is h equal to zero, so I'm saying I don't know anything, let's say that it's zero, so everything constant, or in general to a small constant is the same. Of course, if I have more informative heuristics,

00:31:19 Speaker 1
I have a better clue on what will be the most promising state to expand. And so admissibility is telling me, the heuristic H is admissible. If the true cost of the cheapest path from here to the goal, so I am in S, I will pay something to go to a goal. If this is the best cost, then my heuristic, does not overestimate it. I'm always getting the cost or underestimating it. So I'm not saying, OK, this costs a lot.

00:31:53 Speaker 1
And then I end up looking at the node and seeing that actually the cost was very small. We'll always be the other way around. So of course, H of s is equal to 0 is admissible for any state. And of course, if I know everything, so H of s is, exactly the cost from there to the goal, the best cost, then of course is another admissible one. And we can also see that we have a hierarchy. So this is much better than this, and everything in the middle would be a compromise between the two.

00:32:26 Speaker 1
So if they are admissible, we can also introduce the concept of, domination. So let's say that H1 and H2 are admissible. If H1 is smaller than H2, then H2 is more accurate or dominates H1. Because of course the.

00:32:58 Speaker 1
highest it is, and not larger than the real cost, the better it is, because I am estimating with less error the real best cost. Okay, and the other property is consistency. or also monotonic, we may say, if for every action and for every state and next state, so such that result of S A is S prime.

00:33:33 Speaker 1
So here, just to say, I have a state, I do an action, I go into another state. What happens is that H of S is more or less equal than the cost of S A plus the next heuristic, so H of S prime. So my current estimate, H of S, is more or less equal than the real cost, step cost, plus the next estimate. So it means that I'm not only underestimating any path to the goal,

00:34:08 Speaker 1
but I am underestimating every step. So at each step... Basically, you can write it also like h of s minus h of s prime, smaller or equal than cost, SA. So it means that at each step, the real cost that I'm going to pay is greater or equal than my estimate, my estimate of what will be that specific step cost. So this is just on the full path cost from this node to the end, to a goal. This is for the step cost.

00:34:39 Speaker 1
So step by step, my real cost will be lower bounded by my difference of estimates. And so as intuitive, consistent implies admissible, because I am underestimating each real step cost. And so my estimate overall will be an underestimation of the cost overall.

00:35:16 Speaker 1
Okay, that's questions. Admissibility is intuitive. Consistency, not that much. Like this probably is the most intuitive way to think about it, where basically you can map one-to-one admissibility and consistency and say admissible, the full path from here to down, and the full cost from here to down. Consistent, one step estimate, one step cost. So this probably maps better to remember. And of course it's still greater or equal than zero, and there is another corollary that tells me if it's consistent, also heuristic of the goal.

00:35:47 Speaker 1
is equal to zero. But I mean that's trivial because I'm in the goal, I am also for admissibility, I'm in the goal, I must be bounded by the best cost to reach the goal, but I'm in the goal so the cost is zero. Okay, so if this is clear, we can look at an example.

00:36:28 Speaker 1
OK. So exercise 2.1, very similar to the one of last time. So we just revised the two approaches, and exploit one graph to do that. I will not focus too much on the difference between pre-search and graph search. Actually, we will see that, basically, in this example, they will be equal.

00:36:58 Speaker 1
So if you have the PDF, also follow me, because I may lose some edges.

00:37:08 Speaker 2
OK. So here, of course, I have costs.

00:37:20 Speaker 1
But I also have heuristics and we have said that the heuristic is associated to the state here I have the state space formulation basically So any node maps one-to-one with a state and so I will just write heuristics over the nodes G is the goal state and we will start from A.

00:37:55 Speaker 1
Okay, should be everything. So let's make a few considerations just looking at the graph, In G, H is equal to 0, so may be consistent or admissible. I may already check this if I have a trivial case where the heuristic is not consistent nor admissible, because if this is different from 0, then the heuristic is not. I also have that strange heuristic equal to 100, but that's not strange, because this is an absorbing state. And so if you get here, you never get out. And so it's nice that I will very hardly.

00:38:28 Speaker 1
penalize the fact of reaching that goal. And I will just say, never expand it, because it's useless. So this sounds good. And if you look at our standard quantities, the branching factor here is equal to 3, I guess. Because from here, you have three exiting arrows, and none of the others have more than three arrows around. To reach the goal, it only arrives from here. And then I have this path, which is straightforward. So 1, 2, 3. So D is equal to 3.

00:39:00 Speaker 1
with the a, b, h, g, and here you have a loop. Here you have a loop, so m is infinite. Notice that even if g is the goal state, here I'm just looking at the topology of the graph. I'm not looking at the search problem. So in the graph, this is a loop. And so, in principle, m is infinite. Okay, then let's look at the properties of h,

00:39:31 Speaker 1
since we have them there. And let's ask ourselves, is it admissible? And the answer is, we have to check. How we do that? We have the cost, and we have the heuristic. The graph is relatively small, so we can, for each intermediate node, look at which is the shortest path in terms of cost, and check with the heuristic associated to that node. So for example, let's start from g, which is the easiest. h of g is equal to zero and is it if you look at the definition of admissibility is it smaller or.

00:40:07 Speaker 1
equal than h star of g so the best cost that leads me to g yes because i'm already there and that's zero so this is the trivial one let's look at h so heuristic of h is one is it smaller or equal than the best path cost from h to a goal where it's just this one that is s cost one so this is valid then let's go back to e h of e is two and the best cost that path is e h g because i just.

00:40:39 Speaker 1
can do this so it's two plus one and this is smaller or equal than that one so it's fine, then let's look at v it's heuristic is two and from b i can go to f but then it's absorbing, otherwise i can go up here but it's costly because it's five to one but here it's fast it's one one, so it's smaller or equal than two yes then i have c heuristic is two is it smaller or equal.

00:41:11 Speaker 1
well i can do two things i can go to e but i already pay four or i can go to h but i already pay one and i have seen that the best cause there was two so it's three the best cost from uh from c and this is uh like the best path but if the graph is larger you can also like avoid to do the full path and just say okay the best thing that i found here was two here at least i have to pay one more so i don't have uh i'm greater than two then i have f heuristic is 100 but it's smaller than infinite because here i cannot exit and finally i have a and so heuristic is 5.

00:41:56 Speaker 1
And again, if I go here, I already pay 5. If I go here, I pay 3. And then here, I said that the best was 2, so I pay 5 to the goal. And here, I have 2, and then I'm in the observing. So these will be the best one, which is 5. But it's enough to say that plus 3, so I'm already greater. OK, so is admissible. Is it consistent? And here, we have to suffer a bit more, because to check for consistency, I guess the best way is the last way I wrote it.

00:42:27 Speaker 1
So I have to check for the difference, between any arrow, let's say, and check whether that difference of heuristics, is smaller or equal than the actual cost. And it's always source minus destination. So here, for example, I have a, let's try to see if I can do it directly from this graph. I have a. let's say delta h a difference of heuristic equal to three because it's five minus two and so this is smaller than the actual cost that i paid which is five so indeed it's an this heuristic was an.

00:43:01 Speaker 1
underestimation and i'm happy so we can check this for everything i have from a to b the difference of heuristics is five minus two which is three and it's smaller than the actual cost which is three so here i'm estimating correctly it's equal to the cost and here i have five minus a hundred, so i have minus yes 95 and here it's telling me yes i'm underestimating because it's a much.

00:43:34 Speaker 1
smaller and the point was that one so i'm estimating underestimating a lot but somehow i'm also including what will be like the fact that i will continuously stay here and so i'm estimating to pay a huge amount so i'm approximating like infinite i'm approximating the full path cost, Okay, so let's do that for everything. 2 minus 2, 0. So here I'm estimating to pay nothing. In the end, I pay 1. Okay, but it's consistent. Here, I'm estimating, again, 2 minus 2, and it's consistent. Here it's 2 minus 2, still consistent.

00:44:11 Speaker 1
Then we go down here. It's 2 minus 100, minus 98. Here it's 0 minus 2, so it's minus 2. Here is 0 minus 100, so minus 100. 2 minus 1, again, smaller than 1. Here I'm getting the true cost. Okay, here would be 0 minus 2. And here...

00:44:51 Speaker 1
is zero minus one sorry for the mess here but the point is that we checked everything all of them are smaller equal than the actual cost and so h is consistent so somehow we see that at least empirically that okay the consistency is a stronger property and so it's also most difficult to check because i have to check all the actions the state action pairs while for admissibility it's enough to check the notes so it's stronger i'm happier because i may say that the heuristic.

00:45:26 Speaker 1
is like more regular but it's indeed a stronger condition and more difficult to check okay. i go like this and let's start to look at greedy best first search so here, The evaluation function is equal to the heuristic. So I will not write again like last time, no generation, no extraction is the same. So at no generation, I just generate a goal,

00:45:57 Speaker 1
the node and add it to the frontier, add node extraction, I check whether it is a goal, whether it is in the closed list, otherwise I add it and I expand the node. So let's start with A. And now also in the tree, I'm going to add the heuristic. And I will also say that my frontier is now A. So node expansion, I just have A. I generate the successors, which are B, C, and F.

00:46:32 Speaker 1
And here, the heuristic becomes two, because it's the one of B. For C, I add two again, while for F, I add 100. So now my frontier becomes, this is a tie, so I just follow the lexicographical order, is B, C. And then I have F, which is the most expensive, at least in terms of the heuristic. Notice that I'm not looking at the cost at all. Okay, now I extract B. Successors are E, F, and H.

00:47:03 Speaker 1
Heuristic of E is 2, of F is 100, of H is 1. So now, in the frontier, I'm doing the usual Dijkstra or Breadburster, but now for heuristics. So the one that costs less is 1, so H. Then I have a tie between C and E, that I use less in graphical order. And then I have F. Let's call it like, I don't know, F1. And that's a tie, so I just follow the order when I draw them.

00:47:36 Speaker 1
It's the same. Okay, now I'm in H, and my successor is only G. with heuristic equal to zero. So now my frontier is G, C, E, and the two Fs. Node extraction, I extract G. Is it a goal? Yes, I'm done. So here I did the tree search because I didn't look at the closed list that was what I was extracting. But basically I've extracted A, B, and H. So I don't have multiple extractions of the same node. So graph search will be exactly the same.

00:48:09 Speaker 1
And so intuitively, I'm again, totally disregarding the cost. So in the worst case, in this case, am I complete? So will I find a solution? Well, if it exists, sooner or later, I will find it. And again, it's like depth first, because I have to guarantee that I don't end up in an infinite loop. That's the only case. So for tree search, I need a finite m. For graph search, I need just a finite number of states, exactly as depth first, because this is what happens.

00:48:39 Speaker 1
when I do not have any guarantee on the order in which I am expanded, but I am expanding everything. So sooner or later, I get there to a goal. Of course, not optimal, because again, I'm just following a heuristic, so I may end up with a certain solution that is not optimal at all. And the complexity is v, branching factor. I'm saying I am expanding everything, so b to the power of m, the worst one that we have found,

00:49:09 Speaker 1
like depth first. So in the worst case, I will end up expanding the longest path. And this is tree search. Of course, here you can put the minimum. Between M and the number of states if you do graph search because we got searcher you do not, expand one node more than once so that if the longest path as repeated states here is a little bit more convenient and, Notice that with that first at least I could use one branch here I cannot do that because I have to keep track of the full tree because probably I don't know if you are the rest of you Stand then I have to get back here expand do something. Then it's 11. So here becomes a new optimal one.

00:49:41 Speaker 1
I return back here. I expand it and so on so it's not like that But I do one branch and then I discard it but here I have to store memory everything So space and time is the same and it's like the worst one that we have seen. Okay, So this is just let's say one step to say, okay, we have looked at the two opposites uniform cost I just look at the cost that led me here and at greedy best So I only look at the heuristic that from here leads me to a goal. Let's put them together and so.

00:50:12 Speaker 1
in this case, Let's start. So basically here the evaluation function, is h plus g, so the path cost to the node plus the heuristic here can be s of that node state to the goal. So let's start. Here I have two things to carry. So I have g that is equal to zero because it's the initial state, I have not paid anything so far, and heuristic which is equal to five. So my f is equal to five.

00:50:42 Speaker 1
Okay, not extraction. So here the frontier is just eight. A, I don't write it. I have now b, c, and f. So in b, if you look at the graph, the cost from a to b is equal to three, and heuristic is two, as we have done there. So f here is five. In c, the cost is five, the heuristic is two.

00:51:16 Speaker 1
And so F is 7. In F, the cost is 2, but the heuristic is 100. So F is 102. Okay? So now my frontier is going to be B, C, F, according to F. Okay? Extract B, as before. Now I have E. So the additional cost, B, E, is 5.

00:51:48 Speaker 1
I paid 3, so now G is equal to 8, while H is equal to 2. So my F is 10. Then I have, similarly, F. I have to add 2, so the cost now is 5. The past cost I always report is F, yes, and H is equal to 100. Okay? Let's call them again F1 and F2. And finally, I have H. Path cost is 4, heuristic is 1, so f is 5.

00:52:23 Speaker 1
Okay, now let's look at the frontier. So here I have 7, 102, then 10, 5, and just 5. So h is the best one. Then I have c, e, hc, and then the 2f. d is 1, has a smaller f, so f1 and f2. Okay, and now I'm done again because h is the one that I'm going to expand. g has cost plus 1, so it's 5, the path cost.

00:52:54 Speaker 1
The heuristic is 0, so f is still 5. And so here I will add g, expand it, and find out that it's a goal state. So the tree is the same as before. The main difference is that now we are guaranteed, since H is consistent, that we have followed, not only reached the best cost, so followed the best path, but also that we have expanded the best path for each of the components that we have extracted. And the same would be, if I want to go on.

00:53:26 Speaker 1
in the next step, I say, okay, let's expand, I don't know, the next one is C, so this would be the best path from A to C, because it's the one that I select to be extracted. So, consistency, tells me not only that I reach the goal, and the optimal goal, but also that everything I expand is the shortest path, the best path in terms of cost, to that specific node that I'm going to extract. So let me write here a few properties, the usual ones.

00:53:58 Speaker 1
So it is complete. So yes, because also the other one was complete the if you want to be super precise, Well, of course the branching factor must be finite, but also that was the same, the cost. must be like you if you think about uniform cost search that is a specific case of a star and so also in this case we need that the minimum step cost is positive because otherwise we may end up in an infinite loop with cost zero and.

00:54:28 Speaker 1
Okay, this is fine. What about optimality? We like consider these my conditions as okay, and we say that for three search. It's enough to have admissibility so it's optimal if H is admissible while for graph search we need that H is consistent and. Then, of course, we have what I was saying before.

00:55:00 Speaker 1
So this is the best path cost for each of the nodes that I have extracted. And also, it's also optimally efficient, which basically tells me that I don't have better algorithms, at least better algorithms that do less extractions to reach the goal. And complexity is the one of uniform cost search, because uniform cost search is the worst case of this. So when my heuristic is useless, my consistency heuristic.

00:55:31 Speaker 1
is useless. So let's talk a little bit more about this. So what does it mean that when I have three search, so I am expanding everything, H is enough to be admissible? Well, it means that suppose that I have a certain starting position. Then I go down to B. And I have cost 2 and heuristic 10. Or I go down to, I don't know, C. My cost is again two.

00:56:02 Speaker 1
The heuristic is two. And here, I reach the goal somehow. And here, the other one, which is the optimal one. So if H is not admissible, so I am overestimating the cost, from here to the goal, it may be that I follow a completely different path. Because in this die, I will firstly extract this one. So I will go down here, do something. And let's suppose I never reach 12. And here, the cost is, I don't know, 10. Then I may end up in this goal and not have optimality.

00:56:33 Speaker 1
While if the heuristic does not overestimate the goal, then, of course, these will not be, must be at most, the summation must be at most this cost, which may be smaller than that one. And so I can only end up having a frontier that contains the goal with the optimal cost. And if I have a frontier that contains two different goals, it may be that I have generated another one before the optimal, but at node extraction, for sure, these will be preferred,

00:57:03 Speaker 1
because h is zero here. I am the exact cost, and if this one is better, this is the one that I will open firstly. So it may happen that I generate one node, one goal node, that is not the optimal, but then I will extract firstly the optimal one. And this is the same that happened with uniform cost. Also there we have the extraction, the goal check at extraction, because that's the only moment when I'm sure that this is the best. Okay, while for graph search, the point is that it's not enough to follow a path that contains the goal, but at each extraction, I need to have the best path.

00:57:36 Speaker 1
that led me there, because then I will not expand anymore that specific node. So if I repeat the same, but here I have, I don't know, node D, and here I have another node D. The same thing must happen for any node that I decide to extract. So I need to guarantee that I'm not overestimating the path that led me to any node that I am extracting, so that that is the best path to that node, and so if I have a solution, that is also the best solution, because I only have one chance. I will not expand anymore D. So I need that, I'm reaching, I'm expanding the node D.

00:58:07 Speaker 1
that is with shortest cost, with the smallest cost. So again, the intuition is a little bit more difficult than for admissibility, but the principle is the same. So by doing that step by step, I guarantee that when I extract a certain node, there is no another path that leads me to the same node in the state space with a better cost. So I guess that's it. We have exercise 2.2, which is basically.

00:58:38 Speaker 1
what we did at the beginning with another problem. So again, we have an initial state, a go state. You can do some actions. So try to formalize it by yourself. And here we also have, it's also asked to write down a heuristic. The suggestion, just one final note on this, is that when you, like the problem that you have in the PDF, you have a set of constraints. So very briefly, it tells you, you have your state is composed of some numbers between 100 and 999 and what you can do at each.

00:59:12 Speaker 1
step is to change one digit to start starting from a certain position reaching a certain goal so here if i start with the five six seven i can do six or six seven so add one in the first, uh five six seven sorry four six seven so subtract one in the first or do the same in the second one so five uh seven seven or five five seven and also the same thing on the third so you can add or subtract one in each of the elements and so this may be a relaxation.

00:59:43 Speaker 1
So, because you have other constraints, like some numbers cannot be reached, so, for example, if this number cannot be reached, I cannot do this summation, or you cannot, if you have, like, 109, you cannot add one here, or remove one from here, so you have some constraints, like before, so before we had, okay, I can empty, I can fill, or I can pull, but I have some constraints capacity, and all that stuff, if you remove the constraints, like do a classical relaxation, and you have a solution in that case, which is intuitive, for example, here, suppose I want to reach 777, how, what would be my cost, would be one action, change this to 7, another action, this to 6, another action, again, this to 7, so cost 3, so this is very easy, I can do that, and write a heuristic for everything, like this, and then, of course, this will not probably be the best cost, probably I will pay 4 or 5, because I cannot do 6, 6, 7, but my relaxation suggests me a heuristic, and if this is a relaxation, then the heuristic is consistent, so in that case, I'm guaranteed that then my graph will lead me to an optimal solution.

01:00:37 Speaker 1
So this is just a suggestion. I suggest you also to do the exercise. And if you have some doubts or questions, just write me or ask me next time. OK. Thank you, have a nice day.
