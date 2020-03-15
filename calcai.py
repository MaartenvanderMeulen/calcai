# Inspired by https://deap.readthedocs.io/en/master/examples/gp_symbreg.html
# That example has extensive explanations.
import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0 

pset = gp.PrimitiveSet("MAIN", 3)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='A')
pset.renameArguments(ARG1='B')
pset.renameArguments(ARG2='C')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def rmse(examples, individual):
    # Transform the tree expression in a callable function
    model = toolbox.compile(expr=individual)
    # Evaluate the root of mean squared error (RMSE)
    se = [(model(x[0], x[1], x[2]) - y)**2 for x, y in examples]
    rmse = math.sqrt(sum(se) / len(examples))
    return rmse

def evalSymbReg(examples, individual):    
    penalty = len(str(individual)) / 1000 # Penalty for solutions that are longer than needed
    return rmse(examples, individual) + penalty,

examples = []
for i in range(10):
    x0, x1, x2 = random.randint(1,9), random.randint(1,9), random.randint(1,9)
    y = x0 * (x1 + x2 + 1)
    examples.append(((x0, x1, x2), y))
print("examples", examples)
toolbox.register("evaluate", evalSymbReg, examples)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", numpy.min)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=False)
    optimum = "mul(A, add(B, add(C, 1)))"
    print("solution", str(hof[0]))
    print(f"rmse {rmse(examples, hof[0])}, solution length {len(str(hof[0]))}, optimal solution length {len(optimum)}")
    return pop, log, hof

if __name__ == "__main__":
    main()