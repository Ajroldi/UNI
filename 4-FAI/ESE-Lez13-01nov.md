00:00:29 Speaker 1
Okay, I guess we can start so, One little remark about last time when we were giving a counter example for The constraints of destruction problem where we said, okay ac3 only checks for binary links I put a different here and it had a solution. So this is the the right one you have, two variables that must be different but at the same time they must be equal to another one.

00:01:03 Speaker 1
So of course there is no solution but if you just check for binary constraints then you see that, you do not delete anything so you end up having domains, you have to do search and when you do search you don't find a solution. So today we will finish our exercises about search and what we.

00:01:36 Speaker 1
if we go back at the very beginning of our discussion, we said okay we have our agent and we suppose that the agent is there by, Itself and there's no interaction with other agents and the environment is deterministic single agent as easy as possible and what we're going to see today is a little relaxation of this environment and in particular we will look at a multi agent environment and how are we going to formalize this with a game.

00:02:29 Speaker 1
So again we will look at a very like restricted class of multi agent environments now we will talk about some characteristics but let's say the main formalization today is this one or.

00:02:49 Speaker 2
adversarial search problem so what does it mean it means that we have a search.

00:03:13 Speaker 1
problem still a search problem not I want the agent one which is which is adversarial so in some sense we are going to say as you can already imagine and you've seen in the lecture we have two agents because we are looking at like the easiest configuration that are playing against each other because it's adversarial okay so in this setting.

00:03:48 Speaker 1
As I was mentioning, we will look at a specific class of games, and we are going to look for deterministic, deterministic games with perfect information to players turn taking and zero sum.

00:04:34 Speaker 1
So what does it mean? Of course, they are deterministic. So I do an action. I go to a state as we have seen so far. We will also look at a very simple. Stochastic environment. And in that case, still we have some information, but we have stochasticity, so we will do unexpected value. Perfect information, so when I play, I see everything. I see what I can do, and I see what my opponent can do.

00:05:07 Speaker 1
And the classical example is chess. Okay, I see the board. While in poker or card games where cards are hidden, I don't see the cards of the opponent, so it's not perfect information. To players, so it's me against you, and turn taking. So it's not like continuous when both we can do something, but it's my turn, I do something, then it's your turn, and depending on what I did, you have some actions possible.

00:05:39 Speaker 1
And finally, zero sum, we will formalise it a bit better now, but in general it means that if I win, I'm as much happy as much you are unhappy, and the same for the complementary, if the other player wins, I'm, so basically we say, okay, I win, I lose, if I win 10, you lose 10, it's not that if I win, I get 10, but if I lose, I lose 10, it's not that, it's.

00:06:10 Speaker 1
not that, it's not that,

00:06:40 Speaker 1
Initial state, goal test, or goal function, and step cost function. This is the search problem, the usual one. So here, we said, in general, I have some states, I have a set of actions, I can eventually only do, play some actions, depending on the state, so I have the action function that tells me, in this state, the set of allowed actions is this one,

00:07:14 Speaker 1
the result function that tells you, okay, in this state, I play this action, where I end up, and we have seen only deterministic ones so far, then initial state, where we start, goal test, is this a goal or not, and step cost, how much does it cost to play this action in this state. So what does it change now in our game basically this triple and in this case we have the player function we have the terminal function and the utility function okay so player function this goes from the set of states to one two okay and tells me who is going to play in this state.

00:08:32 Speaker 1
So what does it change now in our game basically this triple and in this state. So it's a turn taking so depending on the state only one of the two players is going to play if the player function returns one then the player that is going to play is one otherwise is two so basically these includes and generalizes the initial state also because okay I will have in s the initial state and is going to tell me it's the turn of player one player two so it's included here then the terminal function again from the set of states.

00:09:07 Speaker 1
To true false so somehow this is in line with the goal function the goal function was telling us okay this state is our goal or not here it's a little bit different so we have another name because in search problems we were just saying okay this is a goal or not here. It's a terminal state but probably I have not finished right because it's a terminal state but I may have to go back in the state of all update my value then go to another terminal state as we will see in a moment and see okay this is more convenient for me so somehow this just doesn't tell you only I finished or not but this is terminal but I may have not finished.

00:09:58 Speaker 1
So this is like a little difference and so we have another name and the utility function that again replaces basically the cost function that goes from the set of state Cartesian product one two to a number. Okay so this tells me in this state player one as a certain reward in a certain state pleasure to as another reward of course you can remove these and put a vector of two elements here it's the same so this replaces the cost function and it's again it's a little different because the step cost was telling me step by step.

00:10:48 Speaker 1
What was the cost to play a certain action in certain state now we only look at terminal states so we don't have like a path cost but but we just have a cost a utility depending on who is the game okay so here we said this is a zero sum game.

00:11:19 Speaker 1
So this means that, for any state, the utility of S player 1 plus the utility of S player 2 is equal to 0. And this is the reason why it's called zero-sum. Because in the end, if I win 10, the opponent loses 10.

00:11:53 Speaker 1
And so we can introduce a couple of conventions. So the first one is that we don't bring with us the entire function or vector, but just we say that the utility in state S... is equal to the utility in state s of player one.

00:12:27 Speaker 1
So we look at the game from the point of view of the first player, and we say, okay, the other one is going to have minus this. Second convention that will allow us to avoid every time to look for the player function is that player one is the max player,

00:13:01 Speaker 1
player two is the main player, and player one starts the game. Okay? we don't care who is player one and player two because you can always put the minus in front of it and, like swap things so the player one is going to maximize the utility function because of course we have defined the utility in terms of the first player and so he's interested in maximizing it.

00:13:38 Speaker 1
okay while the other one is the minus in front of it so he's interested in minimizing the utility okay and of course it starts but if you like swap everything nothing changes. okay just one one more thing what is the solution here we have to find a strategy.

00:14:16 Speaker 1
And what does it mean it means that for every state I'm going to find the best action so the action that maximizes the utility of course of the or minimizing the one of the other one okay so we will see that if we just to spoiler a bit that we have already seen if you use the minimax so we look at the entire tree we can find the best action for each player because we know how to do it.

00:14:49 Speaker 1
Everything and we also know since it's a perfect information game what the other player will have to choose and we suppose that all the agents are rational so the agents do the right move and so we can like map propagate these and find all the best actions for both players. On the other hand, if we prune some branches of our tree, we will end up with the same like optimal path somehow, but not with an entire solution, because we did with we prune some branches.

00:15:25 Speaker 1
We do not look for the best action in that suboptimal branch that we have pruned.

00:15:39 Speaker 2
Questions, doubts? OK, then let's go to the algorithms.

00:16:03 Speaker 1
So first of all. Let's revise a little bit our search tree that in this case we're going to call the entry and its shape is going to be something like this so I'm already writing the game three of exercise four point one so we have basically this.

00:17:06 Speaker 1
Three elements, so the first one means, of course, that this is a max node, so player max, player one, is going to play, the other one is a min node, and the third one is a terminal state. In the terminal state, we end up there, and nobody plays. It's just, okay, if we are here, this is a terminal, so terminal is true, nobody plays, we just collect the utility, okay?

00:17:46 Speaker 1
And the easiest possible algorithm is minimax. So basically, the idea, like for the last time, is to follow... DFS that first so I have my three and I start going in the deepest path and see okay here I collect five as utility then I go back one step I go down here I see two and I say okay I am in player two is better and so on and we propagate this value so of course if we just apply minimax not only the worst case is.

00:18:39 Speaker 1
Branching factor to the power of n but you actually pay branching factor to the power of n in terms of iterations of steps so find complexity because you have to expand the entire tree so as I was mentioning we find the solution so we we see. what is the best thing to do in any state so if for some reason for some approximation one player, does a wrong move you can perform the best one for that's like sub state but.

00:19:15 Speaker 1
of course you're going to pay an exponential cost because to have explore to explore the entire tree and yeah also the you can still assume that it's linear b times m when you look just for the optimal solution otherwise if you want to store the entire solution so all nodes and best action for each of them it's still exponential so if you store everything everything is exponential.

00:19:49 Speaker 1
if you just are fine with the optimal path it's linear as usual okay so let's try, to write a sort of pseudo code so what is max doing in this case well first of all it, initializes this value to minus infinite and then for each successor it updates the value with the.

00:20:32 Speaker 1
max between its current value and the value of the successor okay and what's the idea behind this well i am max player at the beginning i do not know anything so, In the worst case, I will win minus infinite, so I will lose everything.

00:21:03 Speaker 1
Then, while I go through the tree, I see that there is a successor of my node that allows me to win minus two. Okay, minus two is better than minus infinite. And so I update my value and I say, okay, in this game, I can just lose two. I have a utility of minus two. Then I go through the node and I find another one that is plus five, and I'm happy because I'm winning five, and so on.

00:21:39 Speaker 1
And of course, for minPlayer is the complementary. So I initialize the value to plus infinite, because plus infinite means that... This is the utility of max player, so this means that max player wins everything and I lose everything, so it's it's minus infinite from my point of view of main player, okay, so the game ends up with plus infinite, it means that I, but the utility is the utility of player one, so mine is with the minus in front of it, so I lose everything, my utility is minus infinity, okay, and then for each successor, the value is the minimum between the two.

00:22:34 Speaker 1
The value and the value of the successor, which means indeed that, okay, but if I see a branch that leads the game to be plus five, well, better than plus infinite, it means that the other one wins five, I lose five, better than losing everything. and so on. So I am going to be happy if I am mean if I find something negative. Okay, questions? Is this clear? So the only tricky point is this one. So just pay attention that we.

00:23:12 Speaker 1
are always, also if we are in the turn of mean player, we are looking at the entire game from the point of view of the max player. Okay, so let's try to do it here and I will write this again so that we are indeed doing that first. So I start from the root, this is a max player, so the utility is initialized with minus infinite. Okay, then that first,

00:23:47 Speaker 1
so I generate one successor here, it's a mean player. So the value is plus infinite, okay? I don't know any utility yet, so I generate another successor, which is a terminal state with value five. Now, so basically I did this,

00:24:18 Speaker 1
then I have a successor, and for each successor, I'm going to update the value of min, with its value itself, or the successor taking the minimum, so this becomes five. What does it mean? It means that minPlayer, if ends up playing in here, in the worst case is going. to end up with a value of five.

00:24:49 Speaker 1
So the other wins five, and it just loses five and not infinite, okay? But maybe there is something better, so I generate another successor, which is two, and indeed the minimum now, I back-propagate it, the minimum is two, so okay, if I end up here, I have something better, because I'm mean. And finally, the other value is three.

00:25:22 Speaker 1
This is not better, indeed the minimum is still two, so the value of this node is two, okay? So if Max chooses this action, then mean can control the game, and just end up with a utility of two, okay? okay so now i get that because it's that first when i back propagate i have two now i am looking.

00:25:55 Speaker 1
at max so i update the value with the maximum between the value that i initialized and the successor which is always better than losing everything yes plus two okay and so now max knows that in the worst case he's going to win two but he can also try to explore the other possibility so i generate a mean player mean place the first action gets one sorry the value was plus.

00:26:32 Speaker 1
infinite okay one is better so i update then we generate two two is not better for win player, and the same for three. So the value here is 1, we go back, okay, max will never get here, because it's better to win 2. Third one, I generate the first, which is 4, sorry, the value again is plus infinity, 4 is better,

00:27:13 Speaker 1
then I generate 0, for min again it's better, then I generate 5, it's not better, so the value here is 0, and if I go back, 0 for max is not better than 2, and so this is the value of the root, so the value of the game is 2. Max player is going to win 2, okay?

00:27:45 Speaker 1
So of course you need to explore the entire tree so basically you can also just from here see okay this is mean what is the best here to what is the best year one what is the best year zero then I go back and I see two but I mean this is like the algorithmic way to do that and so my solution is here and then here okay so of course this was like a theoretical algorithm.

00:28:42 Speaker 1
Where we see that it is possible to find a solution but of course it's not something practical it's not something that you can apply unless you are playing a very simple game because you need to explore everything to expand the entire tree but you see that you find an entire solution so you are able to find something which is the best for any node where you end up while our second algorithm.

00:29:35 Speaker 1
Is alpha meta pruning. Okay, so it's basically minimax with alphabeta pruning so something that will allow us to prune some branches and so we have to introduce these two quantities alpha and beta and basically alpha is the best so far for max while beta is the best so far for me and our value is going to be between alpha and beta.

00:30:22 Speaker 1
Alpha and beta. So, alpha is saying, okay, in the very bad case, I am max, and I am sure that I will not go worse than this. If you think about there, we had explored the first branch, and in the end, we said, okay, the value here is two. So, I go back, and from the point of view of max player, in the worst case, max player is winning two. And then we can try to explore the other two, and see if it may win more by choosing another action.

00:30:59 Speaker 1
So, after the first branching case, alpha for max player was two. And from this point of view, beta, after the first branch, is plus infinite. it may happen that if Max chooses another branch, then Min loses everything. So it's the best of far from the point of view of the node.

00:31:33 Speaker 1
where I am in. It will be clearer with the example. So the intuition is that one. If I'm here and I see that I am Max, and in the branch that I explored, I win too, then if I see that after one expansion in this branch, in this second branch, Min can lead me to win plus one, zero, minus one, then of course I will not,

00:32:04 Speaker 1
I will never choose this action. So immediately when I see that there is something that will lead me to win less than something else that I have already explored, I can discard this branch, I will never get here. Thank you. Okay, so in the algorithm, in general, we initialize alpha with minus infinite and beta with plus infinite.

00:32:41 Speaker 1
So, if I don't know anything, so far the best solution for max is lose everything, and the best solution for mean is lose everything. Okay, then we look at the max player. Again, we see, okay, its value is minus infinite, as was there. And for every successor, first of all, we also inherit alpha and beta.

00:33:20 Speaker 1
from the parent node so the value in minimax was propagated from a leaf up to the nodes, alpha and beta are only propagated from the parents to the successors and they are not.

00:33:40 Speaker 2
back propagated from the successors to parents okay and now for every successor the value is.

00:33:57 Speaker 1
updated as before and now i check for the value and i see is v greater or equal than beta in this case, Thank you. let me also write the else and then we comment otherwise alpha is updated with the max.

00:34:33 Speaker 1
between alpha and v what does it mean okay first steps as before so in the worst case i lose everything then i have a success successor that suggests me that okay the new value indeed is two, but so i update the value of this node for max player and then i double check if the best solution that we have i have found somewhere for the other player is smaller than.

00:35:08 Speaker 1
this one if that is the case i am sure that min player will never lead me to this node because, min player, has found another solution that leads it to a value of the game that is smaller with respect to the value that I find in this node. So it means that if we end up here, I win more.

00:35:39 Speaker 1
And so minPlayer will never let me win more. And so this branch is useless. On the other hand, if that's not the case, then it's my value that can be updated, because my best solution, if I don't have a best solution for minPlayer somewhere else, and so I will never end here, my best solution improves.

00:36:13 Speaker 1
So the tricky point is this one, that I have to check, if the current value is greater than the best value that I found so far for the other. Because if that's the case, I'm sure that the other will never leave me here. OK. And for min, let's do also that. It's again the complementary. So we initialize the value to be plus infinite.

00:36:46 Speaker 1
We inherit alpha and beta for every successor.

00:36:55 Speaker 2
We update the value with the min between v and the successor.

00:37:12 Speaker 1
And now if the value... is smaller or equal than alpha, we prune. Otherwise, we update beta with the min between beta and the value. Okay? So, exactly the opposite. Again, double-check this point. So, this means that if the value for min in this node is smaller.

00:37:47 Speaker 1
than the best solution that we have found for the game in another branch, for max, then, of course, we are going to prune because max will never lead to this branch where min can win more. Otherwise, we may have found something that is better for min. And so, we update its best value. Okay let's look at it in practice that is more intuitive so let's do the same max player value minus infinite as before alpha minus infinite beta plus infinite okay I initialize as suggested then I generate a successor which is mean so the value here is plus infinite the other two are the same okay we don't switch them alpha minus infinite.

00:39:03 Speaker 1
Infinite beta plus infinite generate first successor which is five I go back, And as for minimax I say five is better than plus infinite for me so I update the value because I'm here and then I say is these values smaller than alpha no but of course the first branch is going to be expanded entirely because we don't have a best solution found so far okay so this is not better.

00:39:46 Speaker 1
I update meta with the minimum which is five between infinite and five so what does it mean it means that I mean player the value here is five and the best solution for me because beta is mean player the best solution that I found so far for me is five for the max player.

00:40:16 Speaker 1
is minus infinite so I may still win everything so if I have a node here that tells minus infinite then of course I choose this and I'm very happy because I'm indeed checking out the best solution for me and the game ends because okay this is the best I cannot do better but okay the next extension is two here we do exactly as minimax so two is better than five yes alpha again this will.

00:40:58 Speaker 1
not be true so we go here the minimum is two and so we update also meta and the last one is three, it's not better for anybody so we just stop here, And the value here is 2. So, exactly as before, now pay attention here. We go back. We said that alpha and beta are inherited from the parents.

00:41:30 Speaker 1
So, I will not update this beta with 2. Okay? Alpha and beta goes from up to down. I will bring back the value, which is 2. And for max, it's better than minus infinite. Okay? So, for every successor, now I have 2. Is this greater than beta? No, because for max, beta is plus infinite. We update. It's alpha.

00:42:03 Speaker 1
Okay? So, from my point of view, so far, if I go there, I win 2. But of course, there may be another branch where I win everything, so let's try. Okay, again, value plus infinite and now alpha and beta are inherited. So alpha becomes two and beta plus infinite.

00:42:34 Speaker 1
Okay, so here we are saying to mean, okay, you may have here one leaf that tells you you lose everything. So there may still be, like for you, it may be a nightmare here, but be also sure that you will never win something smaller than two because we have found a solution for max that leads here.

00:43:07 Speaker 1
And here, max, in the worst case, wins two. So if you are mean, be sure that if you find something that is minus one, the value of this node is going to be smaller than two, because you may end up there. But max will never choose this action, and so we can prove, okay? And indeed, here, the first leaf was plus one,

00:43:37 Speaker 1
so we update the value, because it's better than plus infinite for mean player. But then, the value of this node is smaller, so better for mean than the other solution, and so max will never bring in here, and we can't rule. Because one is smaller than two. So we exploit this.

00:44:09 Speaker 2
Okay? Okay.

00:44:18 Speaker 1
Now, so here I don't have a value. In this sense, my solution will not be an optimal strategy. Because I will not know what is the best action in this node. But I will still have the optimal path. Okay? So let's go to the third one. Again, we inherit. So the value is plus infinite.

00:44:52 Speaker 1
Alpha is two. Beta is plus infinite. We inherit alpha and beta and we initialize value. First successor is 4, better than infinite, and now this is not better than 2 for min, so it may be the case that we end up here. So if here I have something that is, again, greater than 2, Max will choose this node.

00:45:24 Speaker 1
So I have to continue and check whether it's the case. So value is not smaller than alpha. We update beta with the minimum between beta and the value, which is 4, okay? So basically, what does it mean? It means that now I'm sure that the value of the game, because I don't have other successors, is between 2 and 4, because I found a path that leads to 2.

00:45:56 Speaker 1
So here I will never... end up here if the value of this node is smaller than 2 and I found a value that is 4 so I will I mean so I will not choose something worse for me okay so this is telling me that the value of the game now that I am in the last branch is between 2 and 4. Let's generate another one which was 0, now the value is 0 because it's better for me but 0 is smaller than 2 so max will never choose.

00:46:28 Speaker 1
this branch and so we can prove okay so the only thing that we know for these two branches, is that here the value is smaller than one smaller or equal and here the value is smaller or equal than zero and so for this reason here is two here is smaller equal than one here is smaller equal than zero of course max will always go there so we prune this we prune that.

00:46:59 Speaker 1
And the value here is 2, and the solution is this one, okay? So, we are guaranteed to find the same best path, but not a solution in terms of a complete strategy. Okay, so a few remarks. Of course, this is still exponential in the worst case,

00:47:31 Speaker 1
but the average case becomes much better. And, okay, I guess we said everything. Remember always that if you see here beta equal to 2, you don't have to bring this 2 here. Okay? Because otherwise it's 2, 2, you don't update anything.

00:48:01 Speaker 1
But it may be the case that in other branches it's better. Because here, 2 is the best solution found for min from there to down, not from the parents. Because the parents may end up in other branches. Okay? Okay. Then we have exercise 4.2, that if we have some time, but I don't think so,

00:48:37 Speaker 1
so in the end we can try to solve. Otherwise you can do it yourself, is minimax and alpha-beta pruning with a little bit more complex tree.

00:48:56 Speaker 2
Let me get this OK. 

00:49:39 Speaker 1
so let's go directly to exercise four point four and then we go back to four point three because I want to end to complete like the overview of algorithms and what are we still missing is expected minimax. So I've told you, we've discussed max node, min node, terminal state. If you want to have a stochastic game, you're going to include a chance node.

00:50:31 Speaker 1
OK, so a certain stochastic event that basically suggests to you that if you choose a certain action, you have a probability distribution over a set of states. So we are not fully deterministic anymore, but we have a certain probability that that action leads you to that state. another probability that leads to another state and so on.

00:51:02 Speaker 1
So from the point of view of the tree, if we look, for example, at the example, we have something like this. I am MaxPlayer. I have two actions. The first action leads me to a chance node. The second action leads me to another chance node. And what are the possible events? The first one is that the game terminates.

00:51:35 Speaker 1
and the utility is 6 with probability 1.5. Then with another probability, 1.5, we end up in a mean node. And there, one terminal state is 4, the other one is 7. Thank you. Okay, so with reliability 1.5, I win 6.

00:52:06 Speaker 1
With reliability 1.5, min chooses if I win 4 or 7, so of course I win 4. And the same for the other case. So I have first event, that is, we go to a min node that may end up with minus 2 or minus 10, depending on the action of min, while on the other case, with reliability 1.5,

00:52:37 Speaker 1
the game ends and the utility is 8. Okay, so of course you can choose different strategies. Expect the minimax looks at the average. So because you may say, okay, I don't want to risk because you see here in the best case, so if min is not playing, Here I win 8 and here I win 6, so this is going to be better.

00:53:09 Speaker 1
But if I end up in a state where mean chooses, here I still win something, 4. Here I lose a lot, 10. So somehow here you have more risk because you may win more but you also may lose a lot. How do we face risk with expected minimax? Just with the expected value. So what's better if we play the game many times?

00:53:39 Speaker 1
You may also not have 10 euros and so never end there because you are very risk-averse. So you don't want to risk more than 2 euros and so you don't go there. But for us, just the expected value is what matters. So here... I wanted to keep this because basically we need another suggestion, right? So we know how to expand the tree.

00:54:11 Speaker 1
We know what to do when we are here, and we know what to do when we are here. We don't know what to do in the chance nodes, okay? So for chance, how do we propagate values? So we initialize the value to be zero. And for each successor, that now for me is a couple of two elements, the node and the probability.

00:54:54 Speaker 1
So here... I do not just have a successor node, it's a successor node with probability one half. You see that we have put something more in the tree and so we have more values in the algorithms. Here the value is updated with itself plus the probability times the value of the successor. So I'm doing the average with the expected value.

00:55:27 Speaker 1
I start with zero and I say okay probability one times value one plus probability two times value two and so on. Okay, so let's try to do that first with that tree. So we are max, in the worst case our value is minus infinite. Then we generate the first successor, it is a chance node, we initialize it with 0,

00:56:04 Speaker 1
then we generate a successor, it's a leaf, with value 6, so we put it back and we update the value of the chance node, which is probability, this was 1 half, so 1 half times 6, okay? Here we do not update the chance node depending on the value, here we always update the chance node, because that's a node with a probability distribution on it,

00:56:35 Speaker 1
so we have to expand all the values downside and just sum all weighted utilities, okay? Then I go on the other action, like... action of chance so other possibility one half and here you have a mean so the value is plus infinite.

00:57:06 Speaker 1
okay generate first successor it is four better than infinite i generate the other it is seven, no i prefer four so my value here is four and i can back propagate my value and say okay here i have to add one half times four okay so the value here is five. i may end up here and have plus six or here and then mean chooses this leaf.

00:57:42 Speaker 1
and i get four probability is the same so the expected value is five okay, Max is going to be more happy than losing everything. Actually, he knows that in expectation he's going to win the game because it's plus five. Okay, but we also try the other path. So here value is zero and we do exactly the same. So first successor with probability one half is a min node.

00:58:17 Speaker 1
Here the first leaf is minus two. So here value is plus infinite. Minus two is better. So we update it. Then we check for the other, which is minus ten. Much better. So the value for me in here is minus ten, of course. And here we update one half times minus ten.

00:58:49 Speaker 1
Okay, then we expand the other one and we see that it's plus eight. So plus one half times eight and so the value is minus five minus one. So it's the average between eight and minus ten that are the two possibilities with equal, probability. Okay, so it's exactly like minimax, the only point is here just compute the expected value.

00:59:26 Speaker 1
and this is an algorithmic way to do that. So incrementally add the probability times value. Okay, so now here the value is still five. I prefer to go here if I play with expected value. As you have seen, as we commented before, here. In this case, it's also like the best, the risk-averse case, because here I may lose the game.

00:59:59 Speaker 1
Okay? Just one little remark. How does this change if here I have 80? So, I still preserve all the order. If this is without chance notes, you are sure that if here I put min, min will never choose 80, but min will never, at the same time, choose 8. Okay? So, nothing changes if I do not have chance, because you always look for the order of values,

01:00:36 Speaker 1
not for the value itself. Okay? So, if everything remains order in the same way, the best solution is always the same. Okay? But... If I'm doing the expected value, of course, the expected value depends on the values. So here, if I put 80, the expected value becomes 35, and so 35 is better for max than 5.

01:01:11 Speaker 1
So here, the best actions may change depending on the values themselves, not just on the order. While for when there is no expectation, only the order matters.

01:01:31 Speaker 2
Okay? Is this clear? Nice.

01:01:55 Speaker 1
so if you have some questions this is the right moment also if you have some comment. since it's also my last lecture so i will not answer next time.

01:02:33 Speaker 1
okay that's just a convention to say, i take a node which is successor of v you can just put n, Okay, it's just I did the same for expecting minimax you see or minimax without expectation. So it's just that I'm saying the value or a successor of the value. Yeah, you're right. I just brought my successor and so I can just put in there. I was it was intending to stress that that is a successor of V. Yes, yeah, yeah, you can you can say like that. So n is just a node and you are.

01:03:21 Speaker 2
extracting the value of the successor node. Yeah, that's okay. So let's go to exercise four point three.

01:03:48 Speaker 1
And here we are going to see, again, a game starting from a description of the problem in natural language. So we did the same for the search problem. Now we propose a game where you have a pile of seven bricks.

01:04:17 Speaker 2
And your move is split in two, the pile.

01:04:37 Speaker 1
So every time is your turn, you have a pile of seven elements, you split them, then it's the turn of the opponent. that have now two piles that are splitted and can choose one of the piles and split it again. The constraint is that you cannot split into equal piles.

01:05:10 Speaker 1
So if you have a pile made of four bricks, you cannot split it in two plus two. Or two bricks, the game is finished because you cannot split with the equal piles.

01:05:34 Speaker 2
And when all piles...

01:05:48 Speaker 1
have one or two elements you lose so the game finishes and if that's your turn you lose okay. so if we are here i cannot do anything more i have lost okay so let's try to formalize this.

01:06:27 Speaker 1
with the seven elements that we were mentioning starting from the set of states so suggestions.

01:07:01 Speaker 1
yes yes okay so we can say p1 to pn with the pi that has an element between one and seven that tells me how many bricks are in the ith file and we want to be precise the sum from one to n.

01:07:35 Speaker 1
is equal to seven because we have exactly seven bricks okay one little note here I'm not being super formal because we may see, two things I guess. The first one is okay let's just write p1 to p7 because in the worst case I have seven piles made of one brick and so I will start with seven here and all zeros in the others.

01:08:12 Speaker 1
and then while I split I populate also the others with some values but then this will be a little bit more difficult for the other the other ones so we will always need to include the case of not consider the ones with zero values and so on so I'm assuming that this value n is increasing, during the game while I split so at the beginning n is just equal to one I have one single pile.

01:08:45 Speaker 1
with seven elements then after one split n is equal to two and I have two piles and so on, okay so i'm not fully writing this down but that is the idea okay set of actions. so what is my action a split how can i formalize a split well first of all if i have some piles.

01:09:20 Speaker 1
i have to choose which is the file that i want to split so i'm going to have an index that tells me which is the pile i'm looking at and then i'm going to have let's say a number which for me, represents the number of bricks let's say of first pile.

01:09:53 Speaker 1
After split, so I split pile pi, It's going to be split it into two piles the number of elements of the first pile is okay, this fully represents the action because if I have five files and, I split it a pile of five I split it and I say that the first pile is going to have two elements Then the other one of course is going to have three of them.

01:10:24 Speaker 1
Okay, so it's fully represented if I just say what is the number of elements of one of the splitted files? Because I know how many elements I had in the pile at the beginning, okay? so I can just say how what is the file that I split and How many elements are in one of the two new files and just my convention I chose the first one, okay? So here, formally, let's just say that i is between 1 and 7,

01:11:00 Speaker 1
so it's going to represent which is the pile, and i cannot have more than 7 piles, while, or if you want to be more precise, 1 and the cardinality of this set capital P, so that it increases during the game, and P is an element between 1 and 6. Okay, and this is between 1 and 6,

01:11:36 Speaker 1
because i have always to perform a split. So i cannot say i have my pile of 7 elements, and i do nothing. I will always have to do a split, and so at least 6 elements and 1 in the other pile.

01:11:54 Speaker 2
Okay, actions function.

01:12:03 Speaker 1
So, let me be not super formal and just ask myself, the action IP, when is it applicable in a state P1, Pn?

01:12:34 Speaker 1
Okay, so we're not writing the entire function, just the conditions that make an action applicable in a state. It's basically equivalent. It's a little bit less formal because you cannot directly write it as a function, but it's fine. So. Trivial condition. The index must be one of the elements of the pile.

01:13:05 Speaker 1
So if so far I have three piles, I cannot split by four. So the trivial condition is i belongs to the set 1 to n. It's the index of a pile. So now i is happy because I have selected a pile, but I cannot always split it, because if that's a pile of one, I cannot split it.

01:13:38 Speaker 1
But also if that's a pile of two elements. So in these two cases, I cannot split it. In other cases, like a pile of four, I cannot split it in equal piles. So also in that case, I cannot have a pile of two and another pile of two. And how am I going to formalize this? You have different ways. This is the one that came to my mind is that p is in the set one up to floor of pi minus one divided by two.

01:14:26 Speaker 1
So now we comment on this. The point is that, OK, since it's the same, I mean, symmetric actions lead to the same set of files. So if I have a pile of four and so let's do an example, I have a pile of four. I split it into a pile of three, one, and a pile of one, three, okay?

01:14:59 Speaker 1
I don't care because it's the same. I end up having a pile of three and a pile of one. So what I'm saying here with pi divided by two, so one, two, pi divided by two, is that I'm always looking at these k's. So where the first pile is the smallest, okay? I just want to repeat actions that are equivalent.

01:15:33 Speaker 1
So here I say, okay, the elements of the first pile are always going to be from one to half of the pile, and the other one is going to be what remains, okay? But... I am removing one, because I don't want to split it in half. So pi divided by 2 allows me to divide this between 2 and 2.

01:16:06 Speaker 1
If I remove one, I always say, OK, the first one, needs to be half minus 1, at least. And I put the floor, because maybe if it's like this, I have 4 minus 1, 3, divided by 2, and floor is 1. And that's OK, because I can only do 1, 3 here, right?

01:16:40 Speaker 1
Because 2, 2 is not allowed. While if this was 5. Here I have 5 minus 1, 2, divided by 2, 2. Okay, it works because I can have 1 or 2 elements in the first one and 3 or 2 in the second one. 3 or 4, sorry, with 5.

01:17:12 Speaker 2
Okay, is this clear? Or you need an example more. 

01:17:18 Speaker 1
This is fine? So from 1 to half, but I stop 1 before. That's the idea because I cannot split in halves.

01:17:37 Speaker 2
Okay? Okay, then...

01:17:48 Speaker 1
Result function, okay, this is rather easy. So you have a state p1, pn, and an applicable action i, p. What is the result here? The result is another state that starts from p1, goes to pi minus 1, then I have one brick of p elements, another brick of pi minus p elements,

01:18:32 Speaker 1
and then again I continue from pi plus 1 up to pn, okay? So pi represents, the number of bricks that is in that pile so here I have a certain number of bricks in the first pile, up to a number of bricks to the ith minus one when where I didn't touch anything then I split these.

01:19:06 Speaker 1
with p elements in the first so the other one is pi minus it and then from the ith plus one to n I don't do anything okay and now the three new elements that are rather easy in this case so the player function here we can just be not that formal and say this is a third taking game where max starts by convention.

01:19:50 Speaker 1
okay everything that we have just discussed so we don't really need a function if it you can if you want to formalize this okay if the if it's uh even mean place if it's all the max place but this is enough and finally two elements more one is terminal so terminal again refers to a state and this is going to be true or false like a goal function.

01:20:35 Speaker 1
and this is true if pi belongs to one two for any i from one to n. False otherwise, so if all my elements are one or two, I cannot speak anything more okay, and finally utility again is the utility of a state so p1 pn and again I will not be super formal it's plus one if current player is mean minus one if current player is max okay, so if this is my turn and I am.

01:21:46 Speaker 1
Mean I lose. So since it's not specified anything about how much does it cost to win or lose, but these are the two possibilities, so usually win or lose, we are in a zero-sum game, one is plus one and the other one is minus one. There are no ties because at a certain point, one wins or loses. So if min plays, now it's my turn, but I cannot split anything, I lose. I am max, so the utility of the game is minus one. If max plays, and now it's the turn of min, and min cannot do anything,

01:22:22 Speaker 1
min loses, and so the utility of the game is plus one. Because we always look at the game from the point of view of max. And notice that here... Utility, by definition, is only defined on terminal states. So if terminal is true, then there is going to be a utility associated to that state. Otherwise, it's not.

01:22:53 Speaker 1
So we can assume that this is well-defined because we only look at terminal states. So we don't care if there is something in the middle where we do not have utilities, right? So with respect to search problems, here we don't have step costs. We just have the final utility of a terminal state. Okay?

01:23:24 Speaker 1
So I leave it to you to solve the game with minimax and alpha-beta pruning. what i'm going to comment and write is just the game tree so basically minimax is already done.

01:24:12 Speaker 1
okay so i will try here at the blackboard to not make a mess so we start with max player, and in the notes now i'm just writing the search tree so let me write the number of bricks, for each pile in the nodes so here in max node i have one brick one pile of seven bricks.

01:24:45 Speaker 1
okay now possible successors well there are three of them because remember that, in the first pile i can always put something smaller than the alpha so here in the first pile i can put one element two elements or three elements so it's one six and it's going to be a.

01:25:21 Speaker 1
min node otherwise two five it's again a min node or three four what could be another possibility. Here are not equal piles, because it's not possible. I may have 4, 3, 5, 2, 6, 1, but it's exactly the same, with a symmetric. So it's the very same.

01:25:52 Speaker 1
I have two piles, 1 of 6 and 1 of 1. If they are switched, nothing changes. So exploiting how we define this, we just have three branches. OK, now here I have two possibilities. 

01:26:17 Speaker 2
because the first one is 1, and the other one can be 1, 5. Yes, 1, 5. for max or two four i cannot have three three because it will be a split with the same number.

01:26:53 Speaker 1
and i cannot have uh like the other way around because it's symmetric it's the same we don't care, okay then here i can have one one and then split this into one four for mean or one one two three.

01:27:28 Speaker 2
three okay.

01:27:35 Speaker 1
And let's go down, no, let me also do this, here it's simple, because it's 1, I cannot split, 2, I cannot split, 4 can only be 1, 3, because 2, 2 is not allowed, and then I go over, okay? Now here, I can only split between 1, 3, okay? Because it's just 1, 1 split, and then I have 2, 2, which is not possible, and then it's over.

01:28:25 Speaker 1
Okay, I will not repeat that anymore. While here, I'm done, because here, the only split is 1, 1, 2. one, two, okay? Because now they are all splitted, I cannot do anything. So min did the last move. Now here is the turn of max, who loses. So here the value is minus one, right?

01:28:58 Speaker 1
While here, I still have one move, I will do it very badly here, which is one, two, which is terminal. And now min, max played, now it's turn of min. So min loses, okay.

01:29:28 Speaker 1
And also here, the only move allowed is one, two, one, one, two. So it was the turn of mean, now it's max, max loses, okay? Very fastly, the other branches.

01:29:49 Speaker 2
So here, I have 2, 1, 4, and 2, 2, 3.

01:30:05 Speaker 1
The first one becomes 2, 1, 1, 3, which is the only possibility, while the other one is already 2, 2, 1, 2. Last layer was max, it's the turn of mean who loses, value plus 1. Right here, it's 2, 1, 2, mean played.

01:30:35 Speaker 1
So, min wins. Okay? Final branch. I want to end the tree to just give you a suggestion. 3, 1, 3.

01:30:51 Speaker 2
Max 4 to 1. Let me write 3, 4. Okay. 3, 1, 3.

01:31:07 Speaker 1
And here is, I split 3 with 1, 2, 4. Okay, so here I can split 3. I can split 4 in 1, 3 only, because it cannot be 2 and 2. And here I split 3 into 1, 2, and then I have 4 that is not split. Then. 2, 1, 1, 3, okay, here, notice that I end up in the same state whether I split the first 3 or the last one, so since it's the very same, again, I don't care, I just do once, okay, so it's 2, 1, 1, 3, if I split these, it's 3, 1, 1, 2, but I don't care about order, so the numbers are the same, okay, so I just do it once,

01:32:03 Speaker 1
and the same here, so I have 1, 2, 1, 3, it's min, and now we are finished, because this is 2, 1, 2, min played, so min wins, here it's... 1, 2, 1, 1, 2, min played, min wins. Okay, my comment was, look at the branches, okay?

01:32:39 Speaker 1
Here, all this branch ends up with min winning. So the value of this branch is going to be minus 1. If we look at this central branch, and the same applies there, okay? Here, in one sub-branch, max wins. In the other sub-branch, min wins. But who chooses the two sub-branches?

01:33:11 Speaker 1
Min. So also here, the value is going to be minus 1, okay? because I have two branches, in one Min wins, in one Min loses, and who decides which path to follow? Min. So the last real selection is done by Min, okay? All the others are just propagation of the same, because there are not real actions here. And the same applies there, because I have Min who selects between going here and Win,

01:33:41 Speaker 1
or going there and Win or Lose, so it will always end here to be sure that it's going to Win. So here the value is minus one and Min always wins, okay? So if you want to try Alphabeta Pruning or Minimax also here, but probably it will be, if you have to choose, do exercise 4.2. That is better for Alphabeta Pruning, okay?

01:34:16 Speaker 1
So that's it. From next time will be Alberto and we will move to logical agents. If you have any question or something, you can reach me out by email.
