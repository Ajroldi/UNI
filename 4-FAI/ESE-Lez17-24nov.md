00:00:04 Speaker 1
OK, so good morning everybody. My name is Alberto Metelli and I will be the teaching assistant for this second part of the course. On the blackboard, you can find my email, which is albertomaria.metelli at polimi.it, which is the main channel if you need to contact me. So please feel free to drop me an email if you have doubts about the exercise session, about the exercise we solve, and possibly about also other exercises that you may try to solve while preparing for the exam.

00:00:37 Speaker 1
So in the second part of the course, we will have some exercise sessions about several topics. Today we will start with reinforcement learning. Next week, we will have an exercise session about logic and then we will have planning after a few weeks. As I think you're already used to, exercise sessions are organized as follows. In the first part, we will have a basic, say, we will revise the theory a little bit, provide some additional detail,

00:01:08 Speaker 1
while in the second part, we will solve some exercises which are expected to be similar to the ones you will find at the exam. Okay, so today we talk about reinforcement learning. So what's reinforcement learning? Reinforcement learning is a learning paradigm in which we have two entities involved,

00:01:42 Speaker 1
something that we call the environment and something that we call the agent. And the interaction between these two entities works as follows, at every discrete time instant of interaction, we have that the environment provides the agent with an observation of its state, and based on the state, the agent executes an action on the environment, and as an effect of the action, the environment evolves into a new state and provides the agent with a numerical state.

00:02:30 Speaker 1
This is a signal that is called a reward. This reward... defines the goal of the agent because now the agent has the goal of maximizing some notion, of total reward so the cumulative reward that it accumulates while interacting with the environment i will define this notion of total reward formally in a few minutes so it's a goal goal maximize the total reward the mathematical model that we use in order to represent.

00:03:17 Speaker 1
this kind of interaction and in general reinforcement learning problem, is called the markov decision process so what is a markov decision process. is a mathematical model which is made of a tuple of some elements. In particular, we have a set S that is the set of states.

00:03:49 Speaker 1
that the environment can be in. A state is a representation of the situation in which the environment is currently, the current situation of the environment. So set of states, but for us it's going to be a finite set of states, okay? Then we have a capital H, which is the set of actions, again finite. What are the actions? Are the possibilities that the environment.

00:04:25 Speaker 1
has to be executed on the environment, the options that can be executed on the environment. Then we have some other elements, capital P, Capital P is called the transition model and basically describes the way the environment evolves as an effect of the action. Formally, it is a function that takes a state and an action.

00:05:00 Speaker 1
and provides a probability, sorry, a state, an action, and another state, and provides this out with a probability distribution where P of S prime given S and A represents the probability that when my environment is in state S and my agent plays an action A, the environment evolves as an effect of the action into state S prime. Is this clear for everybody?

00:05:35 Speaker 1
Okay, so it's a probability distribution over the next state then. We have r which is the reward function. The reward function r formally is a function that takes a state an action and provides a real number, where, r of, s a represents.

00:06:05 Speaker 1
The reward that the agent gets when the environment is in state s and the agent decides to play actually Okay. Then we have, Gamma Which is called the discount factor? Gamma is a number, That ranges between zero and one. And it basically represents the aptitude that the agent has in giving value to future rewards.

00:06:42 Speaker 1
So if you have a gamma which is close to zero, the environment is more interested in getting reward immediately. While if gamma is close to one, you have an environment that is interested in getting reward in the future. You will understand formally the role of gamma in a few minutes. And finally, we have mu zero, which is called the initial state distribution. Mu zero is a function that takes a state and provides a probability distribution where mu zero of s for state s is the probability that the interaction between the agent and the environment begins in state s.

00:07:35 Speaker 1
Question, so far, should be pretty clear. Why this mathematical framework is called Markov decision process? Because it obeys to what is called the Markov property. The Markov property, Markov property, consists in the fact that, given that, I know the current state of the environment and the current action played by my agent,

00:08:08 Speaker 1
the distribution of the next state does not depend on previous states and previous actions. Okay, so formally, the probability of getting in S prime, say that the time t plus one, I get to a certain S prime, given that I was in state S, I played action A, previously I was in state S t minus one, I played action t minus one. to S0 and A0, so this probability of reaching the next state given the whole history depends.

00:08:43 Speaker 1
just on the current state and the current state. So, said in words, the distribution of the next state depends on the present but is independent of the past. This is called Markov property, and this is encoded in the signature of the transition model of my environment. And notice that here we are in a situation in which we have just one agent interacting with the environment. In basic reinforcement learning there is just.

00:09:16 Speaker 1
one agent involved. Now, this totally describes the way the environment works. Now we need to say how the agent behaves in this environment. And the agent behavior is encoded with a mathematical object that is called the policy. So agent's behavior is encoded by a policy. A policy, which is denoted typically by pi,

00:10:00 Speaker 1
the greek letter, is a probability distribution over the actions given a state. So it's a function that takes a state and an action, provides a probability distribution where pi of a given s, is the probability that my agent plays action a when the environment is in state s. In principle it can be stochastic, so the agent may decide to randomize over actions in a given state. It can be deterministic, but it's not mandatory.

00:10:36 Speaker 1
Just one note on this. Here we are talking about reinforcement learning. So we are talking about a scenario in which we are performing a learning process. What is different between learning and what you have seen previously in the course, which was basically search, also called planning?

00:11:10 Speaker 1
The difference is the following. When you are in search or planning, the way the environment behaves is known to the agent. In particular, for this specific model, when you are in a searching or in a planning problem, the transition model and the reward function are known. Known to who? To whom? To the agent. Differently, when you are in a learning scenario, the environment, the way the environment behaves,

00:11:44 Speaker 1
so the transition model and the reward function are unknown. This is the reason why, as we shall see later on today, when we are in learning, since we don't know how the environment behaves, we will need samples from the environment. So the agent will be in the need of interacting with the environment to collect samples, to try to figure out how the environment behaves in order to find a proper behavior, so a proper policy.

00:12:15 Speaker 1
So it's important to distinguish a learning scenario versus a searching or planning scenario. Questions so far? Okay, so now we have all the ingredients to formally define what we mean with total reward, which is going to be what the agent will maximize, will optimize in a reinforcement learning problem. So this informal notion of total reward is described by a formal.

00:12:53 Speaker 2
notion that is called expected discounted cumulative reward, which is given by this.

00:13:21 Speaker 1
following quantity, you take the summation, over all time instants in which the agent interacts with the environment that in our model is going to be a never-ending interaction so we will have a series with the time variable that takes values in the discrete numbers so from t going to zero up to infinity of what of.

00:13:52 Speaker 1
the reward function that the agent gets when the environment is in state s and plays action at and this is going to be weighted by a weight factor which depends on the discount factor in particular the reward that the agent gets at the time instant t is going to be weighted by discount factor raised to the power t and then since this is in general.

00:14:23 Speaker 1
A random quantity, you take its expected value, and you get a number. I'm going to describe this again in a moment. So let's try to match this expected discounted cumulative reward, each of these words, into the formula. So we have reward that you see here, okay? You're not going to consider a single reward, you're going to consider a cumulative reward.

00:14:54 Speaker 1
So the sum of the rewards you get in your interaction with the environment, cumulative, and this is given by the presence of the summation, okay? Then discounted, At every time step, you're not going to consider the value of the rewards by itself, but you're going to consider it multiplied by discount factor raised to the power of t. And now you see the effect of the discount factor, which is a number between 0 and 1. If gamma is close to 0, you're going to raise a number close to 0 with positive powers, so this is going to decrease very fast as t increases.

00:15:32 Speaker 1
If gamma is close to 1, then you are raising a number close to 1 with exponents which are positive, and so this value is going to stay close to 1, even if it's going to decrease, of course. So you're going to have that future reward will be weighted less. When gamma is small, you're going to have that future reward is going to be weighted more, so close to 1, where gamma takes values which are close to 1. Is this clear to everybody? Please, if you have questions, don't be afraid.

00:16:03 Speaker 1
Raise your hand. Okay. Finally, we have expected. Why expected? Because you know that both the environment and the policy of our agents can be stochastic, okay? So the quantity that you have inside the square brackets is a random variable in general. And you want a number, so you're going to take the expectation. So expectation with respect to the stochasticity is used by the environment by means of this transition model and by the agent.

00:16:38 Speaker 1
By means of its policy. This is the quantity that the agent is interested in optimizing the expected discounted cumulative reward. There are other formulation in which you don't have discount factor. You have an interaction with last for a finite number of steps. They are all alternative, but we're going to stay to this definition. This definition is very important.

00:17:09 Speaker 1
Because, it's the basis of some functions that are used commonly in reinforcement learning. If you now suppose to consider that the interaction with the environment starts in a given state, say state s, so if you assume that the initial state of the environment is s, what you get.

00:17:39 Speaker 1
is a quantity that is called value function of the state s, that we denote with the symbol, v of pi of s. It is the expected discounted cumulative reward that you get when the interaction starts in state s, and the agent is playing a certain policy. Bye. Please. If it is zero, you make the convention that zero raised to the power of zero is one.

00:18:18 Speaker 1
So you're going to consider just the first reward. If it is one, so all weights are equal to one. And in order to give a meaning to this formula, it must be that after a certain point, the reward goes to zero. Otherwise, the series may diverge. So you don't have constraints if gamma is between zero included and one excluded. When gamma is equal to one, it must be the case that you need to enforce that the series converges.

00:18:50 Speaker 1
Otherwise, this gets to infinity or has no value. So value function, value function of a given policy pi is the expected discounted cumulative reward when you start in a given state S. So this is a function from the states to the real numbers.

00:19:20 Speaker 1
There is another important variation of the value function, which is called action value function, or more informally Q function, which is very similar to the previous one, with the only difference that you're not only fixing the state, the initial state, but also the initial action. So action value function, also known as Q function.

00:19:58 Speaker 1
The Q function of a given policy pi is a function that takes a state and an action A, okay, and it represents the expected discounted cumulative reward, so the same stuff as there, but now you are forcing the initial state to be S, but also the first action to be played to be equal to A. Then after the first interaction, you're going to play policy pi, okay, so this is called action value function.

00:20:49 Speaker 1
Why these value functions are important? Because they are giving you a number or numbers to evaluate the quality of a policy, okay? Since, don't forget, that our goal is to maximize the total rewards, so the expected discounted cumulative reward, if I am able to compute these value functions for different policies, I can compare and rank different policies to see which one is better than the other, okay? So now we can arrive to the notion of optimality.

00:21:24 Speaker 1
So when a policy is optimal in reinforcement learning.

00:21:40 Speaker 2
Optimality. Definition.

00:21:50 Speaker 1
The policy, let's call it pi star, is optimal if it attains the maximum possible value of the value function in all the states. So it is optimal if pi star belongs to the arg maximum over all possible policies of v phi of s for every state.

00:22:28 Speaker 1
So this is definition of optimal policy. There exists no other policy which gets higher expected discounted cumulative reward from any state. OK, this is a definition. it is known from the theory of Markov decision processes that such a policy always exists, at least one, possibly more than one, but for sure there exists one optimal policy.

00:22:59 Speaker 1
which is deterministic. So see theorem, there always exists an optimal policy which is deterministic. So the theorem is not obvious at all in the sense that it's not obvious that.

00:23:36 Speaker 1
a policy obeying to this definition exists, first of all, and it's not obvious that such a policy exists, there exists one which is deterministic. okay when i talk about policy i'm referring to objects of that form there so they decide which action to play based on the current state only okay now the point is how would you compute.

00:24:08 Speaker 1
an optimal policy well first of all a couple of symbols the value function of an optimal policy is called the optimal value function okay so we denote it with v star of s this is called the optimal value function it's defined as the value function of an optimal policy.

00:24:42 Speaker 1
Okay, as I said, the optimal policy always exists there. It may be the case that there exists just one optimal policy or more than one. Okay, all optimal policies are equivalent in value. Okay, so all optimal policy obtain the same value function. And we call this value optimal value function. And we can give the same definition for the actual value function. We call this Q star the optimal value function, and it is defined as the value function of any of the optimal policies.

00:25:27 Speaker 1
Questions? Okay, so is this definition clear to everybody. 

00:25:49 Speaker 2
so now how to compute the optimal policy well not the optimal policy one optimal policy if.

00:26:14 Speaker 1
multiple exists so this step is typically done in two phases first you compute the optimal action value function so you compute q star okay that it happens it happens to be easier than computing directly the optimal policy you are going to see why this and then you compute the optimal policy, from q star with the following rule.

00:26:48 Speaker 1
The optimal policy in a state S, I denote it as a function of S only, because I'm going to consider a deterministic one, so a function that takes a state and gives me the action to be played deterministically. The action to be played in state S, the optimal action, when you know the optimal value function, is one of the actions which maximizes the Q function in the given state.

00:27:19 Speaker 1
Okay, so don't be afraid of this. If I give you this function, Q star of S A, and I'm asking you, tell me which is the optimal action to be played in a given state S, you pick state S, and you evaluate this function for all possible actions, you pick the action which attains the maximum value. That is the action to be played in state S.

00:27:50 Speaker 1
Is this clear? Okay, if you have more than one, it means that more than one action attains the maximum, and it means that all these actions are optimal, and you can pick any of them, or you can even randomize over them. Okay, so these are the two steps. First, we compute Q star, and then we derive the optimal policy in this way. When I give you a Q function,

00:28:21 Speaker 1
And I fix a state, the action which attains the maximum in that state is also called the greedy action. OK, so this is going to be the greedy action. In other words, the optimal action is the greedy action with respect to the optimal Q function. Now, it seems that I simply moved the problem from computing by star to computing Q star.

00:28:54 Speaker 1
Now I have to tell you how to compute Q star. And here you have to say, trust me, because we don't have time to go into the details of on why Q star is going to be computed in the way I will explain in a moment. So how to compute Q star. The answer is, using what is called the Bellman equation.

00:29:31 Speaker 1
The Bellman equation is a recursive equation that you can use to compute Q star without knowledge of pi star. And it works as follows. It's the following equation. It tells me that Q star in a given S-A pair is computed as follows. You take the immediate reward, then you sum gamma expectation over the next state of the probability of reaching the next state,

00:30:11 Speaker 1
given that you were in state S and action A, and then you take the maximum. over the actions of Q star of S prime A prime. I'm going to comment in a detailed way this equation in a moment. What is important to note for now is the following. The Bellman equation is an equation that allows me to compute Q star, but it is a recursive equation, because you see that in order to compute Q star in a given state action pair SA,

00:30:45 Speaker 1
you will evaluate the same Q star in another state action pair, possibly in more than one state action pair. Now let me try to give you the intuition behind this equation. So remember what Q star is representing. It represents the expected discounted, cumulative reward that you get starting from state A.

00:31:15 Speaker 1
From state S, sorry, playing action A, and then from that moment on, playing the optimal policy. Okay, this is the meaning of Q star. So let's try to match this definition into the formula. You are in state S and action A. So at the beginning, you're going to get the immediate reward. Okay, immediate reward.

00:31:46 Speaker 1
Then, from that moment on, you're going to play the optimal policy. But now you are in state S, you played action A, you're going to land to another state because the environment is evolving. How the environment is evolving according to the transition model. Okay, so you're going to take expectation over the next state. according to the transition model to understand with which probability, any of the possible next state is going to be reached and this is exactly this portion so this is expectation over the next.

00:32:22 Speaker 2
state next state is s prime of course since.

00:32:39 Speaker 1
you are in the next state you are in the next time instant of the interaction what you are going to get in terms of cumulative reward from that point on it's going to be discounted by gamma discount factor okay because you have done a step discount. And now, which is the quantity of which you're going to take the expectation, the quantity you're going to take the expectation on is basically the expected cumulative discounted reward of the optimal policy that you're going to get starting from the next state.

00:33:14 Speaker 1
And this is exactly that quantity. This is the expected discounted cumulative reward from S prime on. OK, this is intuition behind the Bellman equation.

00:33:45 Speaker 1
This maximum here, you can also write it as V star of S prime. OK. You realize it because of the definition of the optimal policy. The optimal policy attains the maximum of the Q function. Okay, so its value function is precisely this stuff.

00:34:09 Speaker 2
Okay, questions. 

00:34:20 Speaker 1
Of course, I'm unable to give you the full details behind the, say, how this equation is derived. I hope to have, say, provided to you a little bit of the intuition. Okay, now we're ready to see the first and the only algorithm, the first and only reinforcement learning algorithm that we see in the course, which is called Q-learning. Yes?

00:34:53 Speaker 1
Yes, no, it's the same because you see that the probability that the summation is over S prime, okay, which is the only argument of V star, okay, and if you use Q star, you're going to averaging terms of the form max over A prime of Q star, so again, it's a function of S prime only, because V star is actually equal to max over action of Q star always, okay.

00:35:38 Speaker 1
So the algorithm that we see is called Q-learning. And it is an algorithm that allows us to get an estimate of Q-star. Notice that, remember the distinction that we have done between learning and search, okay? In learning, we don't know the environment, so you cannot directly apply the Bellman equation in a learning scenario, because you cannot take expectation of Rp, you don't know P, okay?

00:36:12 Speaker 1
So when you are in learning, and this is precisely the scenario Q-learning is addressing, you don't know P, and what you have to do, basically, basically whenever you want to build a learning algorithm, is to replace expectation over distribution that you don't know with sample means. So Q-learning is nothing but a version of the Bellman equation made using samples,

00:36:42 Speaker 1
and not using expectations. And it works as follows, so you can see Q-learning if you want as a sample-based version of the Bellman equation. It works as follows. Whenever you experience in your environment a transition from a state S plane action A landing to state S' and getting a reward R,

00:37:21 Speaker 1
you will update your estimate of the Q-function in the following way. You take a convex combination between the old value of the Q-function. using coefficient 1 minus alpha, plus alpha multiplied by the reward that you see, plus the maximum over actions A prime of your old estimate of the Q function evaluated in the next state.

00:38:13 Speaker 1
So let's try to give a meaning to this update rule. This is the update rule. So first of all, you have a convex combination between two numbers because you are taking this number multiplied by one minus alpha and this other number multiplied by alpha and alpha is a value between zero and one, which is called the learning rate.

00:38:47 Speaker 1
It modulates how fast you want to forget about your old estimate of the Q function, the larger, the more weight you are giving to the term related to the new sample. OK, then the convex combination is done between what? Between the old estimate of the Q function, old estimate, and this quantity here that is called say it's called the temporal difference target.

00:39:20 Speaker 1
It's not important if you don't remember it. What is important to note is that that quantity is strictly related to the Bellman equation. Let's try to match it. We have the immediate reward R, and there you have the immediate reward R, plus gamma, plus gamma, which that I was forgetting. Here you have gamma. You don't have the expectation over the next state because you are in learning, you don't know the transition model, but you are replacing, you are surrogating that expectation with a single sample estimate because now you are evaluating the maximum of the Q function, not in expectation over the next state, but in the value of the only state S prime that you have experienced in your transition.

00:40:11 Speaker 1
OK, so replacing expectation with a single sample estimate, OK, and then the maximum of the Q function over action is the same that you had there. So you can look at this at this point here as indeed the sample based version of the Bellman equation that you are going to progressively integrate in the estimate of the Q function test to this incremental update rule.

00:40:43 Speaker 1
Questions? Now of course you can memorize this one and apply it blindly. I think it's important that you are aware of what is behind this update rule. So the bellman equation which comes from the definition of value function and back of decision process. Now some of you may wonder but is this estimate.

00:41:15 Speaker 1
actually a good estimate? It's gonna be converging somehow to the true value of the q function? The answer is yes. Under some conditions that I'm not going to explain in detail, your estimate of the q function converges to the true value of the q star in probability one. In the limit of the infinite samples, OK, so if you collect the infinite number of transitions under some properties that are related to how you explore the environment, your estimate is going to converge to the true Q function with probability one.

00:42:07 Speaker 2
Questions? Yes.

00:42:12 Speaker 1
No, during learning, you are not using the greedy policy. During learning, you must use a policy that is guaranteed to explore all the actions. This is one of the conditions that you need for convergence. Let me try to give you the intuition. If you never play an action, you don't know if that action is good or not. So you will need to explore all the actions in order to realize which are the best ones and ultimately which is the optimal one.

00:42:45 Speaker 1
So, the execution of this update rule is performed by interacting with the environment with a policy, which is called an exploratory policy, that tries to more or less explore all the actions. That you can improve it, you can make it converge to the greater one, but you need this condition.

00:43:10 Speaker 2
Other questions. 

00:43:19 Speaker 1
Let's try to give, let's try to see an example of application of Q-learning. This is an exercise which is taken from an exam, which is this one, August 30, 2022.

00:43:52 Speaker 1
This exercise, there is no number, is the exercise on reinforcement learning. So the text is the following. Consider the grid environment depicted below, in which black cells are obstacles, S represents the starting cell, and G represents the goal cell. So you have this, grid environment, which is a three by three grid environment, in which you have that the.

00:44:27 Speaker 1
central cell is an obstacle. Cells are numbered like this. Then you have a starting cell S11, you have a goal cell G. So cells are identified using a pair R and C, row and column. For example, 2, 1 identifies the cells below the starting cell. An agent must learn how to reach the goal cell.

00:45:03 Speaker 1
G, using Q-learning, the agent can move in the four directions. So, the possible actions are the four directions. You can move up, you can move right, you can move down, and you can move left. The agent can occupy any empty cell, the starting cell, the goal cell. The agent is not allowed to move towards an obstacle, okay, or outside the environment.

00:45:34 Speaker 1
Okay, you cannot move outside the environment. The agent is trained using Q-learning with discount factor, gamma, equal to one-half. it's a very low discount factor typically to give you an idea, discount factors are in the order of 0.9 0.99 0.999, okay is a reasonable discount factor for a real problem for the exercise to allow you to make calculations, there is a simpler value and using a learning rate alpha.

00:46:08 Speaker 1
equal to one over four the reward function the attention returns one when the agent reaches the goal cell, okay so if you are here you play action down, you get reward one okay because you have reached the goal cell, if you are here you play action say right you're gonna get reward zero so reward is always zero apart from the cases in which playing a.

00:46:38 Speaker 1
certain action makes you arrive to the goal to the goal cell, and the environment is totally deterministic, okay finally say let's take note of this reward one when reach goal as next state finally the q.

00:47:11 Speaker 1
table that i'm going to explain in a moment what is is initialized with all zeros so what is the q table the q table is nothing but a tabular representation of the q function since we will always consider cases in which the number of states and the number of action is finite the number of entries of our q function is finite and we can represent it.

00:47:43 Speaker 1
using a table so the q table. This is a table where we have as many rows as the number of states. So here we have states and as many columns as the number of actions. So here we have actions which are up, right, down, left.

00:48:44 Speaker 1
which are our states we have state one one we have state one two one three two one i'm not representing two is not a state because the agent is never allowed to be here.

00:49:14 Speaker 1
okay so it makes no sense to represent two two as a state so two three instead three one three two. And similarly, there is no need to represent free-free, which is the goal state in my table, because when I arrive here, the interaction with the environment stops.

00:49:46 Speaker 1
OK, so there is no action that I have to perform in the goal state. OK, so there is no need to represent it in my Q table. So this is the Q table and the text is telling me that is initialized all to zero, with all zeros. But before that, let's observe that some entries of this table can be discarded, because when you are in 1-1, you cannot go up.

00:50:17 Speaker 1
OK, so this action makes no sense. As well, you cannot go right, left, sorry. In 1-2, you cannot go up, as well as in 1-3. In 1-2, you cannot go down because there is the obstacle, and in 1-3, you cannot go right, exactly. In 2-1, you cannot go right because there is the obstacle, and in 2-3, you cannot go left for the same reason.

00:50:51 Speaker 1
In 3-1, you cannot go down, and you cannot go left, exactly. And in 3-2, you cannot go up because there is the obstacle, and you cannot go down, okay? So these cells have to be discontinued, yes. In 2-1, we cannot go left, yes, thank you.

00:51:29 Speaker 1
And also in 2-3, you cannot go right, okay? By the way, there are only two action available in every state. So all meaningful cells are initialized to zero.

00:52:02 Speaker 1
First question of the exercise. In the first training episode, the agent will start in state S and perform the following action. So this is question one. The agent will start in state S and perform the following actions, right? Right, down, down. Therefore, reaching the goal and receiving reward one.

00:52:34 Speaker 1
Okay, right, right, down, down. What are the values in the Q table after this first episode? So the question is, apply Q learning in the way we have seen in this episode of interaction, which is an episode, is a sequence of interaction with the environment ending at a certain point, in this case, when we reach the goal. We have to apply Q learning with this episode. So for doing this, we need to repeatedly apply this update rule using alpha one half,

00:53:08 Speaker 1
one fourth, and gamma one half, considering that kind of interaction. So at the beginning, the first step, the first transition that we experience is the transition from state one one, so the starting state, playing action right. Blending to state 1, 2 and receiving reward 0, right?

00:53:40 Speaker 1
So this is state S. This is the action that I play. This is state S prime. This is the reward that I get. All the pieces of information that are needed in order to apply my update rule. So this is the tuple S, A, S prime, R. So let's apply it. Q of S, 1, 1, given action right, gets updated with what?

00:54:11 Speaker 1
With 1 minus alpha, the same Q function, its old value, plus alpha, multiplied by what? The immediate reward that I get, zero plus gamma maximum over all actions, A prime, of the old estimate of the Q function in the next state, S prime. So one, two, action, A prime.

00:54:49 Speaker 1
Do you agree on this? It's nothing but matching the update rule with the specific representation of states and actions. What's the value that we get here? If we replace numbers, so 1 minus alpha, okay, 1 minus 1 fourth, Q of 1, 1, right, Q, I have to query my Q table in 1, 1, right, zero, okay, zero.

00:55:23 Speaker 1
Then, plus alpha, one-fourth, zero plus gamma, maximum over the action of my Q function in state one-two, taking the maximum over the actions. In state one-two, I have to consider this line and take the maximum value of this row, which is zero. Okay? So, zero again. In the end, I get zero. So, I'm updating my Q table one-one-right, replacing zero with zero.

00:55:55 Speaker 1
Okay? This is the first step of application of Q learning. Let's do it for the next one. What is the next transition? The next transition is, now I am in state one-two, I'm playing action right, and I'm landing in state one-three and getting a reward zero. Okay? So now I have q of one two right that gets value one minus alpha q of one two right plus alpha zero plus gamma maximum over a prime of q now one three a prime.

00:56:48 Speaker 1
And again, I realize that this value is zero because the q of one two right, q of one two right is zero. Then I have zero plus gamma. I have to consider the maximum value of the entry for a row one three. In one three, everything is zero. So this is zero. This is zero. And again, I get zero. So I am updating this zero with another zero.

00:57:21 Speaker 1
See you. Now I think you are understanding that until we perform the update that is related to the transition that leads the state to the gold state, we are going to update zeros with zeros, okay? So let's write it down, the next one. So next one is I am in state , I'm going down, I am arriving to state , and I'm getting the reward zero. So again, Q of 1,3 down gets.

00:58:09 Speaker 1
value 1 minus alpha Q of 1,3 down plus alpha zero plus gamma maximum over a prime. of Q of 2, 3, A prime. Again, Q of 1, 3 down is 0, and Q of 2, 3 for all the actions, they are all 0. So again, I am updating 0 with another 0.

00:58:41 Speaker 1
1, 3 down. And now we have the actual interesting update, which is the one related to the transition that from state 2, 3 going down leads me to the goal state, say G, getting reward equal to 1. Okay, so now what is happening?

00:59:12 Speaker 1
I have Q of 2, 3. Let me do it. say here here so q of two three down gets value one minus alpha q of two three down.

00:59:48 Speaker 1
plus alpha pay attention immediate reward one plus nothing because i have reached a state state after which actually in which no other action is allowed so in this case you have to add nothing when you are in what is called the terminal state so a state after which no other action is allowed you simply consider here the immediate reward.

01:00:22 Speaker 1
so if you now do the calculations this is zero and here you get actually alpha which is precisely equal to one four so now you update q of two three down with one over four okay so you see that.

01:00:59 Speaker 1
All applications of Q-learning in all transitions but the last one basically are ineffective because you start with a Q-table initialized all to zero, okay, and then you basically update computed stuff, which is always zero. In the end, when you arrive in the last transition, you're going to get the positive reward, the plus one reward, and here there is nothing to be added, no gamma maximum over next state, because there is no next state to be considered, okay?

01:01:34 Speaker 1
So when you're considering a transition that leads you to a terminal state like the goal one, your update, that in general is written like this, lacks this part. So this plus gamma maximum. Because it's impossible to define this part. Questions? Okay, now the second question of the exercise is telling us to do another iteration of Q-learning,

01:02:12 Speaker 1
starting from this table that we have updated with the same episode. Okay, so suppose that the agent, is again experiencing the same episode. What do you expect? I don't care about actually numbers, but what do you expect in terms of cells that are going to take values different from zero in another application? Yes, exactly. So your colleague is saying that now what is going to happen is that.

01:02:49 Speaker 1
there will be another cell taking value greater than zero and it is precisely cell one three down why because the update then we're gonna do it for one three down, we'll be considering in this portion of the update the estimate of the q function in the next state so two three yes with the maximum overactions and now two three has an entry because of the previous update with value one four so this one four gets propagated back to one three okay.

01:03:23 Speaker 1
then if you do another iteration with another episode you're gonna propagate back this positive value up to one two and then up to one one okay. This is the precise phenomenon that happens when you apply a reinforcement learning algorithm to problems like this, in which you have a reward which is always zero apart from your goal. At every iteration that reaches the goal, that reward gets propagated back. Let's do it.

01:04:00 Speaker 1
I'll be pretty fast because now we have understood the rule. So now Q writes. will maintain value 0. Why? Because to apply Q-learning, I will be considering the immediate reward, which is 0, the Q-function in , but in , the maximum over-action remains 0. So Q remains equal to 0.

01:04:37 Speaker 1
Then Q for doing this update, I will be considering the old value of Q , which is 0, and then the immediate reward, which is 0, and the Q value in the next state. So what is the next state? The next state is , but has all values equal to 0. So again, Q will take value 0.

01:05:13 Speaker 1
Let us now consider the two updates which are of interest to us, which are the update from 1-3 down to 2-3, getting reward 0. So now in this update I will have Q1-3 down takes value 1-α Q1-3 down plus α immediate reward 0 plus γ maximum over actions.

01:05:58 Speaker 1
Of Q2-3, α prime. Now Q1-3 down is 0. the immediate reward is zero but now notice the phenomenon that i was commenting before you have to consider the maximum value entry for row two three row two three has a maximum value one four okay so now this takes value alpha gamma, times one four okay uh i don't remember okay so one four one.

01:06:42 Speaker 1
four and one two one over thirty two okay. finally one two three going, Down reaching the goal state, getting reward 1, we lead to the update Q down gets value 1 minus alpha, Q down plus alpha immediate reward 1 plus nothing. So 1 minus alpha times Q down,

01:07:26 Speaker 1
now this value is not zero because we have to take this one, 1 over 4 plus alpha.

01:07:43 Speaker 2
So 3 over 4 times 1 over 4 plus 1 over 4 is 7.

01:07:58 Speaker 1
Over 16, OK, if you continue in this update, you will see that this value of this cell, so two, three down is going to progressively increase until reaching what until reaching the true value of the Q function in that state and action pair, which is going to be one.

01:08:28 Speaker 2
So the immediate reward. OK, questions. So let's update the table. So one, three down.

01:08:54 Speaker 1
is 1 over 32, and this takes value 7 over 6 here. So now, last question of the exercise. Suppose you continue to train the agent until the Q table converges. What would be the final value in the table? There is also a suggestion. The values at convergence are the expected discounted cumulative rewards. So now we have to fill this table with the true value of the Q function.

01:09:30 Speaker 1
So in principle, when you are asked an exercise, so you have a question like this, to compute the true value of the Q function, so Q star, you should apply the Bellman equation recursively. But in this simple exercise, and in general in all the exercises, that you will find in exams, the computation of Q-star can be done directly by inspection.

01:10:02 Speaker 1
Okay, why? Because it's simple to realize that the optimal policy in this example is basically, when you are in this in cell 2-3 you have to go down, when you are in cell 3-2 you have to go left, when you are in cell 3-1 you have to go left, when you are in cell 1-3 you have to go down, here you have to go right, here you have to go down, and here it's indifferent if you go down.

01:10:38 Speaker 1
or right, okay? So there is no policy that attains a higher expected cumulative discounted reward, right? Expected is even not necessary because everything is deterministic. So now you have to compute the Q-values, the q function of this policy which is the optimal one and you can do basically in a pretty simple way, so let's start from say state three two so instead three two if you go right you immediately reach.

01:11:22 Speaker 1
the call state and so you're gonna get reward immediately reward one and then the process stops so q star in three two it's gonna be one okay then what happens if you are in three two and you go left if you are in three two and you go left you take reward you get reward zero right.

01:11:53 Speaker 1
And now you are in 3-1, okay? From this state, you have to play the optimal policy by definition of Q star. So you are going to go right, getting again reward zero, going back to state 3-2, and then here you have to go right again. You will get reward one, and then you will arrive to the goal. Do you agree with me? Because here we are computing Q star of 3-2 left.

01:12:30 Speaker 1
Okay, so you're forced to go to the left in the first action. So what is the expected cumulative discounted reward that you have in this case? You have zero times gamma raised to the power zero, plus zero times gamma raised to the power one, plus one. times gamma raised to the power 2, okay? So in the end, you will have gamma raised to the power 2.

01:13:00 Speaker 1
Are you fine with this? Now, by symmetry, we can easily compute the values of the Q function in 2-3, because in 2-3, if you immediately go down, you reach the goal immediately. If you go up, you will have two steps needed to reach the goal. So you are in a scenario precisely like this one. So you will get gamma squared, okay?

01:13:37 Speaker 1
We can generalize this reasoning because... in this precise example then you can of course replace gamma with the actual value and get the numbers in this precise example the optimal q function is for a given state induction is gamma raised to the number of steps needed to reach the goal minus one okay in three two going.

01:14:10 Speaker 1
right you have gamma raised to the number of steps needed to reach the goal minus one one step to reach the goal minus one zero here three two left which are the number of steps needed to reach the goal the first one two three steps minus one okay so in general for this specific example q state action is gamma.

01:14:43 Speaker 1
it's just for this example, raised to number of steps to reach the goal minus one. So here you will have gamma, here gamma raised to the power of three. You will always have that in every row the exponent will have a difference equal to two.

01:15:16 Speaker 1
Then in two one two, say one three, one three if you go down you have gamma, otherwise gamma cubed. In two one two one going down you will have gamma. square otherwise gamma four raised to the power four and the same in one two. One two right gamma square otherwise gamma four.

01:15:49 Speaker 1
and finally in the initial state regardless the actions, you will have what gamma four. So one two.

01:16:05 Speaker 2
three four yes. Okay if I have made the calculation correctly which is not obvious, indeed it's not, I should be free.

01:16:45 Speaker 1
So in general, the kind of exercise you will have in this, the kind of requests when you are asked to compute the Q star are pretty simple. So you can basically realize the optimal policy by inspection and then compute its value, its expected cumulative discounted reward. Questions. 

01:17:16 Speaker 2
Okay, so let's do another one.

01:17:52 Speaker 1
Consider the following decision-making problem. An agent is in a three-by-three grid, again, and the agent can move in the four directions. Yes. Maybe I made a calculation mistake.

01:18:24 Speaker 1
Let's see. Two, one. You mean which one, this one. 

01:18:38 Speaker 2
Three, one.

01:18:43 Speaker 1
So, three, one. You are here.

01:18:46 Speaker 3
You go, your issue is for which action. 

01:19:05 Speaker 1
Both, okay, so first step, you go here, you get zero, so gamma raised to the power of zero times zero, plus second step, you go here, you get one, times gamma raised to the power of one, and so you have gamma, okay? Okay, now, we are in another 3D3 grid, and we have the same four actions, so going up, going down, going left, going right, but we have an additional action, which is stay still, okay?

01:19:44 Speaker 1
Okay, so the agent can decide to remain in the current cell, okay, and I will denote it with this empty circle. Again, we are allowed to perform all the actions, provided that we don't go outside our grid. The interaction starts at the lower left cell, okay? So this is the starting cell in our interaction with the environment, okay? Like the S there. And then we have that this cell here, so upper right cell, is a terminal state, like the goal. Okay, terminal state. If we arrive there, the interaction stops.

01:20:16 Speaker 1
The immediate reward is represented inside the grid. So when the agent is in state, say, lower left, it gets reward zero. When it is in this state, it gets reward zero, okay? Then when it is in this state, it gets reward minus one. In the central state, it gets reward minus 10. In all other states, it gets reward zero. In the terminal state, it gets reward two. Okay? Now, here, it's slightly different, and the text is pretty clear on this. You get the reward when you are in a state, not when your action leads you to another state.

01:20:47 Speaker 1
Okay? Because in the previous example, you get reward one not when you are in the goal, but when you are in another state and you perform an action that leads you to the goal. Okay? Here, you get the reward represented here when you are in this state. Okay? Now, we have the following question. This is the question that I will write it down. For which values of discount factor gamma in 0, 1, the optimal policy consists in staying in the initial state forever?

01:21:24 Speaker 1
So, here. try to reason on this question first of all does it make sense to use a question given the rewards.

01:21:48 Speaker 2
that are represented in the grid yes because at your site we didn't want to move because.

01:22:06 Speaker 1
exactly so it makes sense as a question because you want to maximize the cumulative reward right, the only positive reward that you have is in the terminal cell up there okay but to reach it you must pass through cells with negative reward okay so intuitively then we are going to formalize it if you are an agent with a small discount factor you care a lot about the immediate reward you don't care that much about the future rewards so getting a minus one passing through one of these two cells it's going to cost a lot for you okay much more compared to the reward plus two.

01:22:37 Speaker 1
that you will be able to get in the future if you have an agent with a high discount factor instead you're going to value a lot the plus two okay almost in the same you're going to wait it almost in the same way as you wait the minus ones and so it's going to be convenient for you to reach that terminal cell so the question basically is telling you to compare the value functions of two different policies the policy say pi still which in state let me number the state 0 1 2 0 1 2 let me state 0 0.

01:23:12 Speaker 1
It's telling me to stay still, okay, compared to the policy which, from state , arrives to the terminal state by either doing this path or this other equivalent path, okay? There is no point of considering any other paths. Of course, I will never pass through the Selbig-Meinstein, it's always inconvenient, okay? So the only two policies to be considered are always stay here, as the test is prescribing, or go to the terminal state in either this or this other way. So the other policy I will consider is if I go, which in state is telling.

01:23:49 Speaker 1
me to, say, go up, then in state is telling me to go up, in state is telling, me to go right, and in state is telling me to go right. Then it's totally irrelevant how these two policies are defined in other states. Let's compute their value functions. So V of pi still in state 0, 0 in the initial state is going to be 0, right?

01:24:20 Speaker 1
You remain in that state forever. You're going to get 0 forever. Do you agree? In pi go, evaluated in 0, 0, everything is deterministic here. So when we are in 0, 0, we get 0 multiplied by gamma raised to the power of 0. Then we go up. Sorry, this is like this. Sorry for the rotation.

01:24:51 Speaker 1
So we are in 0, 0. Here we go up. We arrive in 1, 0. In 1, 0, we get reward minus 1 multiplied by gamma raised to the power of 1. Then we go up again. We arrive in 2, 0. We get 0 multiplied by gamma raised to the power of 2. Then we go right. We get 0 multiplied by gamma raised to the power of 3. Then we go right again. We arrive in 2, 2. We get 2 multiplied by gamma raised to the power of 4. Do you agree? So we have minus gamma plus gamma four.

01:25:27 Speaker 1
And so now we need to assess whether for which values of gamma v phi still in zero zero, is convenient, so it's greater than v phi go, in zero zero. Which is if and only if zero is greater than minus gamma plus two gamma raised to the power four.

01:25:49 Speaker 2
Yes. 

01:25:50 Speaker 1
No, it's a terminal state. As soon as you reach it, stop. So now if we do the calculation, we realize that we have to solve two gamma raised to the power three, smaller, than one. so if gamma is smaller than one over root three over say let's do it like this one half raised to.

01:26:25 Speaker 1
the power one third okay it's not that important the value what is important is the sign of the inequality the inequality is gamma smaller than something so for small values of gamma you have that it is convenient to remain in the initial state and this is compliant with our intuition if your agent has a small discount factor the plus two reward that you're going to get in the future counts less than the case in which you have a high discount factor okay questions.

01:27:02 Speaker 1
Okay, so to summarize, exercises of reinforcement learning typically are the following. Either you are given with a simple situation in which you are required to do some consideration, like this one, or computing Q star, or other possible exercises that you may find is you have some sequences of state action and rewards, and you have to apply Q learning. Okay, so the second case is just doing a calculation. In the first one, you have to understand a little bit what is behind in order to draw the proper conclusions. Okay, so if there are no questions, I'm stopping, and see you next week.
