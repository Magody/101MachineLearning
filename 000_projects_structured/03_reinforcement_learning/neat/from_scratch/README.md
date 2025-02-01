# Experimental notes
Experiment notes:
- 

Axiomas:
- Hyperparameter configuration actually is very important for every problem.
- probability to mutate links and nodes affect a lot, even if in the final result the genome hast the same number of connections/nodes. 
- Remove stale species can extint all and reset in bad circunstances
- Increment population and generations is not working, wasn't working due to remove stales
- Is important to keep the best individuals though generations without changes as a copy in case that a mutation fails.
- More individuals and more generations have to result in constant or better results respect to low metrics, otherwise the algorithm is failing
- Very sensible to initialization: We have to test the code with atleast 10 differents seed to see if it's really not performing well.
- We need high change of nodes and links ( 0.1,0.2) otherwise random will fluctuate a LOT the results.
- we have to adjust C1,C2,C3 with every experiment due to different nature
- C3 is very important to create a lot of species respect to threshold. Increasing it usually creates more species. Is useful to separate stagnated species.
- Separating species should improve performance or keep constant respect to low species implementation.
- A lot of disables can involve in create excessive neurons. Keep this probabilities relatively low.
- To a good implementation of neat be good, have to get good scores with at least 10 different random seed numbers (10 different environment initializations).
- probability_mutate_connections_weight have to be high, and step for mutation have to be related to the activation function
- Low number of maximum neurons need more population and more generations. And a step a little lower.
- A crucial part to avoid stagnation is: mutate the mutation_rates with 50% to be multiplied to A (A in range 0.9->0.99) or B (B in range 1.01->1.1)
- We can embed genes to inherit, for instance: mutation_rates are genes in this implementation.
- Stagnant species control can decrease general performance due to these childs mutated can generate a new good solution after N (20 < N < 50) stagnant generations. But if number of stagnant generations are really high, we should remove them.
- Protecting stagnant species is important. With minimum 2 individuals.
- enough population is better than a lot. Prefer to increase epochs than population
To do
- Optimice stagnation: In rare cases when the fitness of the entire population does not improve for more than 20 generations. Refocus the search in promising parts.
- Penalize fitnesses using network size.
- species threshold dynamic, to increase species when needed

