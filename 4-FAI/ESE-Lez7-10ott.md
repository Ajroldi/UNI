00:00:08 Speaker 1
Okay, I guess we can start. So, good morning, everyone. Can you hear me? Nice. Okay. So, my name is Paolo Bonetti. I'm a postdoc researcher at AIB. I will be teaching assistant for next Friday. And I work in the group of Professor Estelli that probably you have met or you will meet around sometime. And also together with the next teaching assistant, who will be Professor Metelli. So, the email is the main point to contact me. And eventually, if we need a meeting in person, I am in person.

00:00:41 Speaker 1
Okay, so today we will start going back to the first lectures that you had. So, we will very, very briefly mention like agent environment, search problem, and so on. And we will focus on an informed search strategies. So let's start from the beginning. You have met an agent, and you have talked a bit about what is an agent, how are we going to define it. So we talked about thinking as human, acting as human,

00:01:13 Speaker 1
and so on. And then you end up defining the agent, this, as a rational agent. And so talking about rational agent, you said, OK, this is an agent that is acting rationally. And so what does it mean? That we look at the environment, we look at the action that the agent is performing on the environment,

00:01:46 Speaker 1
and not really on how it is thinking or if it is thinking similarly to what I would have thought. So a little bit more formally, a rational agent, means that you choose.

00:02:01 Speaker 2
An action maximizing the let's include also the expected value of a certain performance measure And let's also have given its knowledge of the environment.

00:02:36 Speaker 1
so throughout the different algorithm and things that we will look we will somehow deal with actions and, A rational agent is the one that selects them focusing on a certain performance measure on a certain environment So now we will like sharp, More precisely the performance measure and different possibilities that you have for the agent, uh, but in general, so here you include what you have defined like utility-based goal-based because i'm not talking about.

00:03:11 Speaker 1
How it is choosing the action so to pursue which goal but just saying that I have actions and I have some performance that I want to maximize so. Giving a bit more of a keyword for the environment very very briefly. So today and for these lectures we will focus on a fully observable environment environment.

00:03:37 Speaker 2
single agent deterministic static and discrete.

00:03:54 Speaker 1
So basically, for what we will look today and in the next few lectures, we don't really have to do the expected value, because everything is deterministic, the environment doesn't change, I see everything. And so we will just look on how to search for a goal or how to perform the action. And in particular, we will focus on goal-based data. So what does it mean? It means that I am in a starting state, I will end up in a certain goal state, I will have some paths that go there. Some of them are with small cost, some of them are smaller, longer, and we will see how, what we find depending on the search strategy that we consider.

00:04:33 Speaker 1
And so I have a starting position and initial state and I have a goal state. And now we will formalize it a bit better in a search problem. But this performance measure, so let's try not to confuse goal-based and like the other famous one, so utility-based. So these are, or at least can be, two different things. So if we consider a goal-based agent, my objective is to reach the goal, and I can maximize my performance measure in terms of the path cost.

00:05:05 Speaker 1
So I still have the cost, the step cost and the path cost when I have a solution, and that is my measure of what is my performance, and I have a goal, because I just want to start from here and go there. I don't care about the intermediate states. On the other hand, if I have a utility base, I may have just a graph and go and be obliged to go around the graph, so at each time step, I have to do one step, and depending on the step that I do, I earn some different quantities, so in that case, I'm utility-based, because I don't have a goal state. I just have some states, I have to go around them,

00:05:35 Speaker 1
and so in this case, I am utility-based, I just want to maximize my performance, so small cost over large reward. Okay, is this clear. 

00:05:49 Speaker 2
Okay, so.

00:05:51 Speaker 1
Like this, and so let's focus on the main topic of the day, which is a search problem. Can you see also if it's not in full light up here? So a search problem, Is characterized by we can say three sets and four functions So I have the set of states the set of actions the initial state, the actions function the result function the world function and.

00:06:32 Speaker 1
Step off function. So s is just, The set of states, let's write it down that we said we will focus on a discrete set also, Not necessarily finite, So I have some states like a, b, and so on. 

00:06:55 Speaker 2
and I have a set of actions, so like a0, a1, and so on.

00:07:07 Speaker 1
And of course, also an initial position. Also this one, in a general problem, may be just a stochastic quantity, so I may say, okay, I have some nodes, and with a certain probability I start from here, with another probability I start from there. In this case, we will assume that it's just one state at zero, belonging to S, so the deterministic starting point. Then, let's focus on the functions.

00:07:30 Speaker 2
The actions function goes from the states. 

00:07:37 Speaker 1
to the actions, and actions, yes, actions of S is a subset of the set of state, of the set of actions, sorry. But not necessarily necessarily the entire set of actions. So it's the set. I mean this function returns a set which is the set of, this of, applicable actions in state s and, again, what does it mean is that if I have like a grid work and.

00:08:12 Speaker 1
I start from here or I'm in this current. This is my state s1 so here actions of s1 is, Is equal to a because I can go like up left right
