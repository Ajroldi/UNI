00:00:03 Speaker 1
Okay, so good morning everybody. So today we talk about propositional logic. So first of all, why we care about propositional logic in an artificial intelligence course? Well, the main point is that you may have agents that determine their decisions based on a logical inference. And so propositional logic is the simplest system, the simplest logical system that you can design in order to perform some logical inferences. There are more complex logic systems, for sure, like logic and so on, but for the sake of the course, we will consider just propositional logic.

00:00:43 Speaker 1
So what is propositional logic? It's first of all a formal language, and as any formal language, it's characterized by a syntax. The syntax in a formal language determines the form of the sentences in your language, so informality represents the form of sentences. For instance, syntax tells you that the formula if p then q is a well-formed formula, so a formula which obeys to the syntax rules,

00:01:15 Speaker 1
those formulas are called well-formed formulas, w, f, while for instance the formula, implication pq is not a well-formed formula. I will not discuss in detail the syntax of propositional logic because you already know it. The second element which characterizes a formal language is semantics. Semantics is the set of rules that allows you to provide, to assign a truth value to sentences. So how to assign a truth value to sentences.

00:01:55 Speaker 1
we will discuss semantics more in detail so in order to give a truth value to sentences we need to introduce the notion of word so it is a word in a propositional logic language it is a function, w which takes a propositional symbol and provides and assigns to it a new value say true or false where this calligraphic p is the set of propositional symbols that you are considering in your language set of propositional symbols for instance consider a propositional.

00:02:37 Speaker 1
language where you have just two propositional symbols say p and q a possible word, is a function that assigns to propositional symbol p, false, a possible word may assign to propositional symbol q the truth value false. Another word, say w1, may assign to p the truth value 2 and to q the truth value false. The importance of the notion of word lies in the fact that you, starting from a word, you can assign.

00:03:09 Speaker 1
using truth tables the truth value to any well-formed formula. So for instance, if you are in this language in which you have p and q, you can, discuss the truth value of the well-formed formula if p then q. So for instance, if both of them, p and q, are false, you have that the implication happens to be true. If the, antecedent p is false but the consequence, sorry q is true then the implication happens to be true the only scenario in which the implication.

00:03:40 Speaker 1
is forcing is when the antecedent p is true and the consequence is false and finally both are true the implication is true so given this truth table and you have all of them for all connectors so implication and or not double implication starting from a word you can assign a true value to this formula so for instance in word w you have that if p then q happens to be two okay because you look at this row p is false q is false implication is true instead in w1 the implication happens to be false because you are in this line here where the p is true but q is false so this is the only.

00:04:15 Speaker 1
scenario which the implication happens to be false given definition given phi a well-formed formula in your language we say that a word is a model for this formula if it makes it true so phi w word. W is a model for phi if it happens that W of phi. Okay, so we call a model any word that makes a formula.

00:04:49 Speaker 1
Okay, this is to establish a common notation. Questions so far? Okay, now we come to the main, the first important concept that we will see today, which is the notion of logical entailment. This is something which is about semantics.

00:05:19 Speaker 1
I will give the definition and then comment it. So definition. Consider a set of well-formed formulas, phi 1. up to phi n. These are well-formed formulas, and they belong to this set, kv, that we call knowledge base. So these are well-formed formulas. Then, let's consider psi, another well-formed formula. We say that the knowledge base logically entails formula phi,

00:05:52 Speaker 1
and the symbol for logical entailment is this one, okay? Logic, logical entailment. We say that the knowledge base logically entails psi, if and only if every model of all the formulas of phi one up to phi n in the knowledge base is also a model of formula psi. So if and only if every model of all phi one up to phi n.

00:06:24 Speaker 1
is a model. of psi. This means informally that whenever all the formulas in the knowledge base are true, all the formulas of psi must be true. From an AI perspective, if you are dealing with a logic-based agent, this means that your knowledge base will represent the knowledge that your agent has on the environment it is interacting with, and the agent has to decide, has to assess whether another property of the environment is true or not. If logical entailment holds, it means that the.

00:06:56 Speaker 1
knowledge that my agent has on the environment is enough to decide that that property psi is true. So this is the interpretation of this notion, which is purely from logic in the AI perspective. Is this definition clear, everybody? It's very important. Because today what we are going to do is to assess logical entailment. So the problems that we will address today are the following. Given the knowledge base psi, assess.

00:07:28 Speaker 1
if the knowledge base logically entails sum. Okay, these are the problems, this is the problem that we will address. How? Basically we will consider two classes of approaches. The first one is model checking. Model checking is a class of approaches which are based on semantics. For instance, an example of model checking that we are not going to use today are truth tables. We will see instead another approach which is DP-LL,

00:08:00 Speaker 1
which is nothing but a search approach, search in the sense that you know, which uses a specific kind of search, backtracking search, coupled with some specific heuristics to try to find a model for a formula. Okay, we will see this later. Search plus heuristics. The second class of approaches that we can consider are called logical inference. also called the theorem proving these are purely syntactic approaches they will not look at truth.

00:08:42 Speaker 1
values they will use just syntactic rules and they are based on the following idea the application of, inference rules inference rules are nothing but typical graphic rules that works as follows, in a general case suppose that you have some formulas phi 1 up to phi m, an inference rule tells you that if in your knowledge base you have formulas phi 1 up to phi m you can add to your knowledge base another formula say phi m plus one okay and the inference.

00:09:12 Speaker 1
rule is written like this on the top of this horizontal line you place the formulas that you must have in your knowledge base on the bottom you place the formula that you can add to your knowledge base based on this inferential example modus points. is an inference rule that tells you that if in your knowledge base you have a formula of the form if phi then psi, and you have another formula of the form phi, you can add to your knowledge base the formula psi. So if in your knowledge base you have phi, you have also if phi then psi,

00:09:42 Speaker 1
you can add psi. This is an inference rule. Notice that we are not talking about truth values, we're just talking about syntax. Then of course you can invent the inference rule that you prefer, but not all of them have the same properties. It's pretty intuitive to get convinced that you should use some inference rule which preserves some connection with semantics. And the properties of the inference algorithms, of the logical inference approaches, which connect syntax to semantics are the following. But before that,

00:10:13 Speaker 1
we need to introduce a definition which is called syntactic consequence. definition let a be an inference algorithm which is an inference algorithm nothing but a set of.

00:10:43 Speaker 1
inference rules okay let's call it a then given a knowledge base knowledge base and the formula, phi psi sorry we say that formula psi is a syntactic consequence through the inference algorithm a of the knowledge base and the symbol for syntactic consequences is this one it's like logical entailment but with just one horizontal line syntactic consequence.

00:11:15 Speaker 1
if and only if there exists a sequence of applications. of the inference rules of a that from the knowledge base leads to psi. So syntactic.

00:11:50 Speaker 1
consequence means that you can start from your knowledge base, apply the inference rule of your inference algorithm if you arrive to psi without an application of this inference rule. If there exists an application of this inference rule which allows you to arrive to psi, then psi is a syntactic consequence. It's a purely syntactic definition. It's a definition which is about syntax. Okay? So notice that we're not talking about models, we're just talking about application of inference rules. Questions so far? so now we need to link these two definitions so what are the properties of an inference algorithm.

00:12:21 Speaker 1
that allow me to derive some syntactic consequences which also have some relation to models so some relation to logical entertainment and the properties of the inference algorithm which allow us to make this connection are soundness and completeness soundness so these are properties of inference algorithms the properties of inference algorithms so first soundness so let's a be an inference.

00:13:05 Speaker 1
algorithm we say that the inference algorithm is sound if and only if, I will tell you in words, then we will write the formula. If and only if whatever you can derive syntactically, it's also logically entailed, okay? So if and only if, if you can derive, using the inference rule of your algorithm, the formula psi, so psi is a syntactic consequence, this statement's about syntax, then formula psi is also logically entailed by the knowledge base, okay?

00:13:38 Speaker 1
Notice that here we are establishing a relation between syntax and semantics. The inference algorithm is complete. if and only if, basically the opposite. Whatever is logically entailed can be derived syntactically using your inference algorithm. So if from the knowledge base, you can logically entail formula psi, then formula psi can also be derived by the application of the inference rules of your algorithm, okay? Questions, this is very important. This property must be clear.

00:14:12 Speaker 1
so question for you is modus ponens sound as an inference rule so consider an inference algorithm that has only modus ponens as inference rule is this inference algorithm sound what do you think who is voting for yes who is voting for no okay it's sound okay because look at the truth tables, if you have that there is a model for phi and for phi and psi okay that model is also a model for psi okay so it's sound is it complete it's not okay because there are formulas which can be.

00:14:44 Speaker 1
derived but they sorry there are formulas that are logically entailed but cannot be derived by modus ponens so now we look at our first logical inference algorithm well first and only inference algorithm for today at least which is resolution resolution. so the solution is based on the following result which allows you to assess logic entailment.

00:15:20 Speaker 1
which is the following it's called the reputation theorem reputation reputation theorem tells you that if you have to check whether a knowledge base logically entails a formula psi okay you can equivalently check so this whole if and only if suppose that your knowledge base is made of the formulas phi 1 up to phi n logic entailment holds if and only if the formula phi 1 and phi 2 and so on up to phi n and the denigration of psi is unsatisfiable.

00:15:55 Speaker 1
unsatisfiable means that it admits no model there is no way to make it true, okay refutation theorem tells you this equivalence between logical entailment as and the unsatisfiability of this formula which is obtained by making the conjunction of all formulas in the knowledge base in conjunction with the negation of psi by the way refutation is what you are doing when you are trying to prove something by contradiction you take the hypothesis the knowledge base you negate the thesis and you prove that this is impossible you cannot make it true okay then there are some other concepts that we need to introduce before using resolutions that.

00:16:25 Speaker 1
are the concept of literal a literal is a propositional symbol or the negation of a propositional symbol okay so p is a literal not p is a literal q is a literal not q is a literal, okay then the concept of clause a clause is a disjunction of literals. so l1 or l2 or ln where li are literals. Given these two definitions we can introduce the.

00:17:04 Speaker 1
inference rules which are used by resolution. Resolution is an inference algorithm which uses just one inference rule which is called the resolution rule. The resolution rule tells you the following. Suppose you have two clauses say l1 or l2 or so on up to l say n or p. Let me make this p say explicit so I will say that there is a literal p here. Then suppose you have another clause say l1 prime or l2 prime or lm prime or not p.

00:17:39 Speaker 1
The properties of these two clauses are that they have in common one compositional symbol that in one clause appear positive, so with no negation, and in the other clause it appears negative, so with negation. If you have two clauses with this property, then you can rewrite, add into your knowledge base another clause, which is obtained by the union of the literals of the two clauses, apart from the one which appears with the opposite sign. So you can add the clause L1 or L2 or N or L1 prime or L2 prime or, and so on, LM prime. So basically you remove these two literals, P and not P. Pay attention, just one liter at a time, okay?

00:18:13 Speaker 1
If you have P, if here you have P or Q and here not P or not Q, you can remove just one of them, not both of them in one application, okay? Because you lose, you lose a sound relation otherwise. Okay, is the resolution rule clear? This clause that you obtain as a result is called resolvent. now to apply the solution in an example you must start from formulas which are of this from here so they are clauses and to do this you need to bring your formulas that may include.

00:18:49 Speaker 1
other connectors into what is called the conjunctive normal form c and f conjunctive. normal c and f a formula is in conjunctive normal form if it is written as a conjunction of clauses this and we see how to bring a formula in contact with normal form for an example which is exercise.

00:19:21 Speaker 1
6.2 you will see that this is pretty straightforward you just need to apply some steps in sequence so consider formula 5 which is a if and only if b or c, this is not in conjunctive normal form it's not a conjunction so an end of clauses you need to bring it the steps are the following first elimination of the double conditional okay so double condition can be written as a conjunction of conditionals in the two directions.

00:19:52 Speaker 1
so in this case you write a if a then b or c and if b or c then a okay the double conditional is a split into two conditionals in conjunction second step elimination of single conditional. so you know that a single conditional when you have if psi if i then psi it's logically.

00:20:23 Speaker 1
equivalent to not the antecedent or the consequence okay so here you have the formula. logically equivalent to not A or B or C, and not B or C or A. Okay, you agree? Then, third step, if you have negations which are applied not to propositional symbols but to complex formulas, you have to bring them inside, using the Morgan.

00:20:56 Speaker 1
So push negation inwards, using the Morgan. So your formula is logically equivalent to, here, for this first part of the formula, we already have a disjunction of literals. So this is already a clause. So nothing to do here. Not A or B or C. While here, you have a negation applied to a complex formula. So you have to bring it inside. The negation of this junction is conjunction of negations, the Morgan. Okay? And not B and not C or A.

00:21:29 Speaker 1
so this is a clause this is not a clause because you have some n's and some r so last step for is to distribute this junction over conjunction and so here you have not a or b or c and not b or a i'm distributing the disjunction over the conjunction, and not c or a and now you have a conjunction of one two three disjunction of literals which.

00:22:04 Speaker 1
are clauses so you are now in conjunctive normal form okay so it should be pretty straightforward just the application rules yes yes write the truth table you can write the truth table you realize that this is a logical equivalent the two formulas are logical other questions. Ok? Ora, proviamo ad applicare la risoluzione ad un altro esempio.

00:22:48 Speaker 1
Questo è l'esercizio 6.3. Dobbiamo assaggiare se Φ1 intende logicamente Φ2, dove Φ1 è la formula A o B o Q, e Φ2 è la formula A e B o A e Q. Ok? Quindi, la risoluzione funziona usando il teorema di reputazione. Quindi, invece dell'integrazione logica, che è l'equivalente, dobbiamo assaggiare se la formula Φ1 e non Φ2 è insatisfattibile.

00:23:25 Speaker 1
Quando vuoi applicare una risoluzione a una sensibilità o a una sensibilità, a una formula, devi portarla in forma normale congiuntiva come primo passaggio. Quindi consideriamo questa formula e la portiamo in forma normale congiuntiva. Vediamo che Φ1 è già in forma normale congiuntiva. È una congiunzione di un singolo litro, che è il globo, e del globo. Non Φ2, attenzione, devi considerare non Φ2 qui, ok? Non lo è, perché non Φ2 è la formula seguente. Non è A e B o A e Q.

00:23:56 Speaker 1
Quindi portiamola in forma normale congiuntiva, applicando tutti i passaggi. Non c'è una condizione doppia o una condizionale, quindi passiamo al passaggio 3 immediatamente, portando la negazione all'interno. Quindi De Morgan, negazione di disgiuntiva è congiuntiva di negazioni. Non A e B e non A e Q. Ora, di nuovo, dobbiamo applicare il passaggio 3 un'altra volta, perché la negazione è davanti alla formula complessa. De Morgan, di nuovo, la negazione di congiuntiva è disgiuntiva di negazioni. Non A o non B. and not a or maybe i okay not a or not q okay and now we are in conjunctive normal form.

00:24:39 Speaker 1
okay are you following. because according to refutation theorem if you have to assess if phi 1 logically entails phi 2 you have to assess whether phi 1 which is your knowledge base and not phi 2 which is your thesis is unsatisfiable okay so i have to bring in conjunctive normal form this formula formula which means bringing in conjunctive normal form phi 1 and bringing not phi 2 okay so let's consider the set of clauses that we have here we have one two three and four clauses overall right and so now.

00:25:15 Speaker 1
we can apply the solution rule considering those clauses i will need basically to combine these clauses, and apply resolution rule when it is possible. Okay, so when the two clauses obey to that property. So, first of all, we list down the clauses. First clause is A, this one. Second is B or Q. First is not A or not B. Fourth is not A.

00:25:47 Speaker 1
So now, to apply resolution, you have to combine each pair of clauses, try to see if you can apply resolution rule. If you can, you apply it, and you add the resolvents to your knowledge base. So, one and two, no, you cannot apply resolution rule. You don't have a literal in the negation of the literal. One, three, yes, because you have A and not A. The resolvent happens to be not B, okay? Because you eliminate this A with this not A, and you get not B. So, five. It's not B, which is obtained by combining the formula one and formula three. I keep track of the formulas that I have combined here.

00:26:20 Speaker 1
Then one with four yes, you eliminate A with not A, and you get not Q, one and four. Then two and three, you have B and not B here. So the resolvent is not A or Q. Not A or Q. And we have applied the solution, yes, the solution rule to formulas two and three. Then two and four yes, because you have Q and not Q. And you get the resolvent, not A or B.

00:26:50 Speaker 1
Then three and four, no, you don't have opposite literals. So you completed your first iteration of resolution, and you have added new formulas to your knowledge base, so you have to continue. I will discuss the stopping conditions at the end. So now, the second step of resolution, requires that, pay attention, that you combine the old formulas with the new ones, and the new ones together. So first. 1 with 5, no. 1 with 6, no. 1 with 7, yes, because you have A and not A, you get Q. 1 with 7. 1 with 8, yes, you get B.

00:27:29 Speaker 1
Then, 2 with 5, yes, and you get Q. But Q is already there. Okay? So it makes no sense to add it again. This is a set of clauses, okay, no repetition. So 1, say, 2, sorry, with 5, you can apply the solution, but you get Q, so I will not add Q again. 2 with 6, yes, and you get B, but B is already here. 2 with 7, no, you cannot apply the solution. 2 with 8, no. 3 with 5, no. 3 with 6, no. 3 with 7, no. 3 with 8, yes. You have not B and B here, and you will get not A.

00:28:03 Speaker 1
or not A, which is not A, okay? So, 11, you have not A by applying the solution to 3 and 8, okay? If you apply it, say, blindly, you will get here not A or not A, but it's equivalent to not A, so you can simply write not A. Okay? So three with eight. Then four with five, no. Four with six, no. Four with seven, yes, you get not A again. Four with eight, no. Then you have to combine the new formulas together. Five with six, no. Five with seven, no. Five with eight, you get not A again.

00:28:34 Speaker 1
Six with seven, you get not A. Six with eight, no. Seven with eight, no. Okay? So you have done here. Since you have added other formulas, which is different from the previous one, you have to continue. And again, you will need to combine the old formulas with the new ones and the new ones together. Okay? We're almost done here because now we have one with nine, no. One with ten, no. One with eleven, yes. You have A and not A. And you get what is called the empty clause. Okay? For step 12, you get the empty clause, which is represented by this empty square.

00:29:08 Speaker 1
condition for a solution if you get the empty clause you stop okay so stopping conditions are the following first you get the empty clause and if you get the empty clause it means that the formula you started from is unsatisfiable okay so if it happens to get the empty formula it means that the formula you started from that in our case was phi 1 and not phi 2 is unsatisfiable which.

00:29:41 Speaker 1
by the way by refutation this means that the logical entailment we were trying to assess in the beginning codes okay this is the first stopping condition second you don't add new formulas the other situation is that you try to combine your formulas you don't have new formulas you stop in this case you can conclude that your formula is satisfied. Okay, questions. So please, when you solve these exercises, even if you see that you can combine formulas following, say, a smart ordering that allows you to get the empty clause, you have to proceed algorithmically, combining all the formulas in the way I have described. Yes?

00:30:27 Speaker 1
Would we have solved on the number 9, if we had to do an elimination? No, no. That was what I was telling you. If you see that you can generate the empty formula, say, combining your formulas in a clever way, when we ask you to apply a solution, you have to apply it completely. Okay? So combining all the formulas, you realize this. When you could have realized this? When you are combining the old formula with the new one, but you realize the empty clause before, because when you combine clause one with clause the other. Okay, other questions?

00:31:00 Speaker 1
The fact that if you get the empty clause, the problem is unsatisfiable, and the fact that if you don't add new formulas, the problem is unsatisfiable, the formula is not unsatisfiable, is telling you basically that the resolution is both sound and complete. So resolution is sound and complete, ok? Because whatever you add, if it derives syntactically the empty clause, then you have unsatisfiability and zoological entailment, and this is soundness.

00:31:32 Speaker 1
If you are unable to get the empty clause, then the formula is unsatisfiable and zoological entailment doesn't hold, so this is completeness. Ok, so now let's see an example instead which uses a model checking approach, so DPLM. So notice that in resolution, we were not talking about the truth values, we were just talking about the application of typographic roots, so logical inference.

00:32:03 Speaker 1
Instead, in model checking, we are explicitly looking for a model, so looking for a truth value, truth values to be assigned to our formulas, to our propositional symbols, to make the formula true. So DPLL, these are just the first letters of the surnames of the vectors, so there is no specific meaning. It's a model checking approach, which is based on a search strategy that you know, that is backtracking search.

00:32:38 Speaker 1
So it's a search approach that starts from a node, the root node, generates one successor, then generates the other successor, and so on, until you arrive at the end, and then you go up, you generate another successor, you go up, you generate another successor, and so on. It's like DFS, but you generate just one successor at a time. it is also based on conjunctive normal form so you will need to bring your formulas in conjunctive normal form and then it uses some specific heuristics by the way finding a model for a formula you can always cast it as a constant satisfaction problem okay you have variables.

00:33:10 Speaker 1
propositional symbols you have values to also to be assigned okay so which are the heuristics which are used in dplr of the following first one is pure literals sorry it's pure symbols what is a pure symbol a symbol is pure if it appears always either not negated or negated in all the formulas okay so it's a propositional symbol that appears either always not negated or negative.

00:33:51 Speaker 1
If you have a symbol which satisfies this property, the heuristic is telling you, you have to assign it to value true, if it appears not negated, you have to assign it to false, if it appears negative. So, don't forget the fact that our goal is to find a model for a formula. If you have a symbol which always appears non-negated, there is no point not to assign it to true. Okay? It's the only choice that you have. The second heuristic is the unit clause. A unit clause is a clause made of just one literal.

00:34:24 Speaker 1
Made of one literal. Again, if you have a unit clause, remember that you are in conjunctive normal form, so you are taking a conjunction of clauses, you have a clause which is made of just one literal, assign it to true. Okay? There is no point not to assign it to true if you are going to find a model. Okay? So, these are the formulas. DPLL works as follows. First, you try to apply these heuristics in this order. First, pure symbol, then unit clause. if at a certain point you cannot apply them, so there is no pure symbol, no unit clause, you have no other choice than to branch. So to choose one propositional symbol,

00:34:57 Speaker 1
try to assign it to false and then try to assign it to true. Let's see an example which will for sure clarify all your doubts. I'll go here. It's the same problem as before. We have to assess the logical entailment of these very same formulas, and we're going to do this checking whether this formula is unsatisfiable or not. So if we're able to find the model or not for this formula. But we're going to do this with DPLL. How? So when I apply DPLL, I typically write down a table.

00:35:32 Speaker 1
made of three columns. So first column is assignment, second column is hypnosis. first column is ruled so in the central column i will write down all the clauses that are coming from from my problem so they are a d or q not a or not q and not q okay these are the.

00:36:06 Speaker 1
clauses then dpll tells me that i have to try to apply these heuristics pure symbols do we have a pure symbol so a symbol which either appears always negative or not we don't because a here is not negative here it is negative maybe i forgot something yes sorry my made a mistake here the clauses are a b or q not a or not b and not a or not q. okay so a is not pure not negative negative b is not pure not negative.

00:36:37 Speaker 1
negative q is not pure not negative negative so no uh pure symbols unit closer yes we have a unit closer which is this one made of just with a so we assign a to true remember that you're considering that formulas in conjunctive normal form you have a conjunction of all these clauses you want to make it true and so there is no point not to this unit close must be true so a equal to this is the assignment and the rule that they have applied is a unit closer on a okay then since a has been assigned to true i have this clause becomes true this clause doesn't change b or q this.

00:37:11 Speaker 1
clause is false or not b okay because not a is false now so not b not a or not q not q do you agree on this, i have simply simplified my clauses given that a has been assigned to true so now i have to proceed again trying to apply my heuristics pure symbols i don't have pure symbols if you apply unit clause you cannot generate your symbols in general but we have now some unit clauses which are either not b or not q you have to work with one at a time so let's.

00:37:42 Speaker 1
consider not b i will assign b to false okay because again i have a conjunction of clauses i want to make it true so the only way is to make not b true so b false so here i have true, b is false so i have q not b is true not q remains the same unit clause on not b then again i have no pure symbols but i have two unit clauses let's consider q i will assign q to true.

00:38:18 Speaker 1
Okay. and so now i have completed the assignment and it happens that they have foreclosed three of them are true the last one is false they are in conjunction so the result is false okay so you have found the false meaning that the original formula is unsatisfiable in other words the pll, allows you to find well it's another one that attempts to find the model of your formula you are unable to find the model okay following just the application of the heuristics and so.

00:38:49 Speaker 1
you conclude that no model exists which means by definition of the form is unsatisfiable which means by reputation that the logical containment holds questions now this was an example in which we were able to find to detect unsatisfiability simply applying the heuristics without branching okay it's not always the case let's see an example in which instead we are forced to branch.

00:39:24 Speaker 1
so this is exercise 6.6 and we have the following clauses the clauses are the following p or q.

00:40:00 Speaker 1
p or not q not p or q not p or not r so i will also keep track somewhere see here, of the search tree that we're going to build while assigning, truth values to our propositional symbols. So at the beginning we start with an empty assignment.

00:40:38 Speaker 1
Then we apply our heuristics. Pure symbols. Do we have a pure symbol here? Yes, not r. Not r is a pure symbol because it appears only this time negative. So not r equal to true, so r equal to false. Pure symbol on not r. Which means that I have the assignment r equal to false. So now the situation becomes p or q, p or not q, not p.

00:41:10 Speaker 1
or q. Whenever you apply any of these two heuristics one formula will become true, for sure. Q non è puro. Abbiamo uniti chiude? No. Ok? Quindi, tutti i risultati falliscono. L'unica maniera in cui dobbiamo procedere è a dividere, è a branchare, cercando di assignare prima la falsa a uno dei nostri simboli proposizionali e poi a true. Ho deciso di dividere sul simbolo proposizionale P. Ok? Quindi ora, dividere su P.

00:41:41 Speaker 1
Quindi, prima considero P uguale a falsa, arriviamo all'ultimo e poi considero P uguale a true. Dovete considerare entrambi. Ok? Se arriviamo all'ultimo di una branca, non è il caso che potete sempre concludere. Quindi, ora sto dividendo. Quindi, prima considererò R uguale a falsa e P uguale a falsa. Ok? Quindi ora, con P uguale a falsa, ci otteniamo questo simbolo. Falsa, quindi Q. Falsa, quindi non Q. 2 o Q è vero.

00:42:12 Speaker 1
e qui ci otteniamo true. Ok? Now, we have unit closes, so I assign Q to true, unit close on Q, so here you have R equals false, P equals true, Q equals true. And so you have true, false, true, and true. So it's a conjunction, you have a false, so the result is false.

00:42:42 Speaker 1
But now pay attention. You have just followed one branch. You cannot conclude from that that your formula is unsatisfiable. Because you have just considered one possible assignment, namely the one which tries to assign P equal to false. Okay, so now you have to continue and consider the other branch. So again, split from P. Now consider the case in which P is true. With P equal to true. what you get is the following you get true you get true you get q and you get true and then you get.

00:43:17 Speaker 1
to this now you have pure symbol on q q equals true so true your symbol thank you. you have all two here and so the formula has a model which is namely the assignment here, i equal force q equal true and p equal true and so you cannot conclude that your formula.

00:43:48 Speaker 1
is unsatisfiable is indeed satisfiable okay so pay attention when you are applying the dll and it happens that you have to split on the value of one variable, you follow first one branch if at the end of this branch you have found the model okay you're done if you haven't found the model you have to follow the other branch this is simply search where the goal is find the model okay in search you stop as you find your goal okay questions.

00:44:23 Speaker 1
okay so now we see another example another exercise which comes from an exam which is exam.

00:44:56 Speaker 1
from 2022, July 14. So the text of the exercise is the following. Consider the following knowledge base in propositional logic. You have P. You have if P and Q, then R. You have if S or P, then Q. And you have T. First question. Apply the resolution inference algorithm using the unit resolution strategy to establish whether R is logically maintained by your knowledge base.

00:45:31 Speaker 1
So you have to assess whether formula R is logically maintained by your knowledge base using unit resolution or resolution with unit resolution strategy. What does it mean unit resolution? Unit resolution is a restriction of resolution in which you are allowed to apply this resolvent rule only if at least one of the two clauses is a unit clause. Okay? you see that this is a restriction of resolution meaning that in full resolution you can apply this resolution rule for general clauses in unit resolution only if one of the two clauses.

00:46:03 Speaker 1
satisfies being a unit clause okay so restriction of resolution so we have to bring our formulas in conjunctive normal form in particular we have to bring in conjunctive normal form the formulas in our knowledge base and the negation of the thesis so not r so this already conjunctive normal form this one is not so p p and q then r is logically equivalent to negation of the antecedent of.

00:46:36 Speaker 1
the consequent elimination of conditionals demorgan to bring this inside and now we have a closer. this is not the third formula is not in conjunction as well so s or t then q, elimination of the conditional negation of the antecedent or the consequence the morgan to bring.

00:47:07 Speaker 1
the negation inside not s and not t or q we are not in conjunction yet distribution distributive property of the r over the end so not s or q or n or q and now we have to glue this from this formula then t is already in conjunctive thermal form not r is already in conjunctive normal form so now let's apply resolution with this unit resolution strategy.

00:47:48 Speaker 1
let's list all all our clauses so we have p, we have the single clause which comes from the second formula in the knowledge base, which is not t or not q or r. Then we have two clauses which come from the third formula in our knowledge base, which are not s or q, not t or q. Then we have t, which is the last formula in our knowledge base, and then we have the negation of the thesis, not r. Okay, are you following? Now, let's apply.

00:48:23 Speaker 1
unit resolution. So we have to combine our formulas together, see if we can apply the resolution rule, but under the constraint that one of the two formulas must be, at least one, must be a unit clause. So 1 with 2, yes, because we have p and not p, and p is a unit clause. We get not q or r. So apply 1 with 2. Then 1 with 3, no, 1 with 4, no, 1 with 5, 1 with 6, no. Then, pay attention, 2 with 3. in principle you could apply resolution here because you have q and not q but none of them.

00:48:56 Speaker 1
is unit closed so you cannot apply unit resolution okay so you see the station now so it makes sense to compare two with unit closing so i will not check two and four two and five you cannot apply a solution because you don't have opposite sign literals two and six yes because notar is a unit closer and here you have r so eight is two and six which lead you to not p or not q then three with four makes no sense to check three with five no three with six no four with five yes because here you have t and here you have not t and t is a unit close so nine you get q which is obtained.

00:49:30 Speaker 1
by four and five four and six no five and six no so you are done for the first iteration okay again you are not hitting the stopping condition because you don't have the empty close and you have added new formulas to your knowledge base so you have to continue, you have to combine these formulas the old ones with the new ones and the new ones together so one with seven no one with eight yes and you get not q one with nine no two we've make sense to.

00:50:00 Speaker 1
check only with nine two with nine yes and you get not p or r then three with nine no four with nine no five with seven no five with eight no five with nine no six with seven yes but you would get the not q again six with eight no six with nine no so now we have to consider seven with nine.

00:50:32 Speaker 1
and we get r seven with nine and then eight with nine and you get not not p. again we have to continue so all the formulas these ones with new ones and new ones together, so one with 11 no one with 12 yes but you get r that you have already here one with 14 you get the empty closer okay yes.

00:51:12 Speaker 1
well in 12 2 and 9 2 and 9 choosing the close. so we're getting the empty closer so this means that the original formula is unsatisfiable okay because of reputation it means that the logic entertainment holds well.

00:51:44 Speaker 1
this statement is a little bit i have to comment a little bit more on this to say that because of the fact that we get the empty clause it means that the formula is unsatisfiable so the logical entertainment holds we need to argue about the properties of this inference algorithm which is not resolution unit resolution so what about the properties of unit resolution in terms of, soundness and completeness what can you say about this soundness of unit resolution do you think.

00:52:17 Speaker 1
that unit resolution is sound start from the observation that resolution the full resolution is both sound and complete what do you think who votes for yes sir who was for no does anyone want to try to argue on the soundness yes because the unit resolution will give us our subset, exactly. Il punto è che se si ottengono le luci aperte con la risoluzione unica,

00:52:48 Speaker 1
si potranno sempre ottenere le luci aperte con la risoluzione completa, perché la risoluzione completa ti permette di applicare la regola della risoluzione in un set di casi più grandi. Quindi, è sottile dire sì, perché se dalla mia base di conoscenza e non dalla tesi ottengo le luci aperte con la risoluzione unica, guarda questo simbolo, prendo la mia base di conoscenza in congiunzione con la legazione della tesi utilizzando la risoluzione unica, applicando le regole sintattiche della risoluzione unica, ottengo le luci aperte, ok?

00:53:22 Speaker 1
Poi dalla base di conoscenza in congiunzione con la mia tesi otterrò con la risoluzione completa le luci aperte. Siamo d'accordo su questo. Questo è ciò che stiamo parlando. Ma ora sai che se ottenghi le luci aperte in risoluzione completa, significa che... the formula is unsatisfiable because of the soundness of resolution this is the soundness.

00:53:55 Speaker 1
of resolution okay the formula is unsatisfiable means that it logically entails false okay so, if you combine this chain of implication you get that if you derive the empty closed unit resolution that then your formula is unsatisfiable okay and so you have proven soundness, this way of reasoning based on the fact that if you're given an inference algorithm algorithm and they propose to you a restriction of an inference algorithm you will always preserve soundness if the original difference algorithm is sound okay is this point clear so this supports.

00:54:30 Speaker 1
my statement that since we have obtained the empty close with unit resolution then the logical entailment that we were checking in the beginning holds because of the soundness okay what about completeness.

00:55:02 Speaker 1
is unit resolution complete in your opinion in other words is it possible to find an unsatisfiable formula such that if you apply unit resolution you will not get the empty clause if this is the case it's not complete okay yes exactly if you are able to find a formula in conjunctive normal form made of clauses none of them is a unit clause and the form is unsatisfiable then unit resolution does.

00:55:36 Speaker 1
nothing okay one resolution because of its completeness the one of resolution you will find the empty clause so let's see if such a problem exists well it's pretty simple to find one which is the following p of q and not p or q and p or not q and not p or not q, this is clearly unsatisfiable okay because whatever two value assigned to p or q there will be, just one of them that is true okay so this is unsatisfiable but what does uh what unit resolution.

00:56:08 Speaker 1
does with this formula basically nothing because you will never you are not able to apply the resolution rule on this formula because none of the clauses is a unit clause okay so unit resolution is not complete and there is nothing to argue about from providing a counter example okay and. again this consideration holds in general if i give you a difference algorithm and then i give you a restriction of this inference algorithm you may lose completeness it's not always the case but you may use completeness symmetrically if i give you an inference algorithm and an extension.

00:56:41 Speaker 1
of an inference algorithm so an algorithm which allows you to apply other inference rules compared to the original one you may use soundness but you will preserve completeness okay questions. So the text of the exercise, I didn't follow the precise question, but question one was to apply unit resolution, and we have done it. Second question was to assess, to say whether the fact that you find the empty clause allows you to conclude logical entailment, and we have argued on this based on Southwood-less. And then the last question was, is unit resolution complete in general, and why?

00:57:13 Speaker 1
And we have also answered to this last question. Okay, since we have five bits, even if it is not requested by the exercise, let's try to apply DPLL on this very same problem, okay. 

00:57:50 Speaker 2
Thank you.

00:57:54 Speaker 1
so we have the closest p not p or not q or r not s or r not p or q p not r no sorry not s or q okay. so few symbols do we have few symbols p is not pure q is not pure r is not pure not s is pure okay so pure symbol or not s so s equals false here we have p p or not q or r this becomes true.

00:58:33 Speaker 1
not t or q true sorry let's use true explicitly because we have also the symbol t not r then do we have other pure symbols well t is not pure so no pure symbols but we have unit clauses so, We apply unit clause on P. So P equal to true, so true, this is false or something, we get something, here is true,

00:59:09 Speaker 1
here doesn't change, P and not R. Now, we don't have pure symbols, but we have unit clauses. So let's apply unit clause on T. So T takes value true, and now we have true, not Q or R, true, false or something, something, true, not R.

00:59:40 Speaker 1
Again, we have unit clause on Q, Q equal to true, so we have true. forse qualcosa qualcosa vero non r, quindi anche qui abbiamo un punto fermo su r, quindi r è uguale a vero e ora abbiamo vero e.

01:00:18 Speaker 1
forse quindi siamo arrivati alla fine senza branching solo applicando le euristiche la formula è falsa quindi possiamo concludere che è insatisfattibile perché tutte le lezioni erano mandatorie in qualche senso le euristiche ti dicono quali lezioni sono mandatorie se vuoi trovare una moda ok quindi senza branching e alla fine puoi concludere che è insatisfattibile e quindi di nuovo otteniamo gli stessi risultati di una soluzione di unità, domande. so if there are no questions i'm stopping and we will see not next week but but the week after the.

01:00:52 Speaker 1
next one and we will have an exercise session about planning have a nice weekend.

01:01:11 Speaker 2
people who.

01:02:18 Speaker 3
Grazie mille
