from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import random
import numpy as np

NUM_OF_RANDOM_SCP = 10
POPULATION_SIZE = 50
UNIVERSE_SIZE = 100
NUM_OF_SUBSETS = 250
NUM_OF_GENERATIONS = 1000
UNIVERSE = list(range(1,UNIVERSE_SIZE+1))
UNIVERSE_SET = set(UNIVERSE)
MUTATION_PROB = 0.25
count = 0

fitnessValues = [[-1e9 for _ in range(NUM_OF_GENERATIONS)] for _ in range(NUM_OF_RANDOM_SCP)]

def generatePopulation(popSize: int, listOfSubsets: list) -> list:
    totalSubsetsInList = len(listOfSubsets)
    population = []
    for _ in range(popSize):
        numOfSubsetsToBeIncluded = random.randint(1, totalSubsetsInList)
        indexSelected = random.sample(range(totalSubsetsInList), numOfSubsetsToBeIncluded)
        curState = [0 for _ in range(totalSubsetsInList)]
        for ind in indexSelected:
            curState[ind] = 1
        population.append(curState)
    return population

def fitnessValue(state: list, listOfSubsets: list) -> int:
    n = len(state)
    # print(n)
    subsetsIncluded = np.count_nonzero(np.array(state))
    curSet = set([])
    for i in range(n):
        # print(i)
        if state[i] == 1:
            for num in listOfSubsets[i]:
                curSet.add(num)
    if(curSet != UNIVERSE_SET):
        return 5 + len(curSet)/5
    global count
    count += 1
    # print(subsetsIncluded)
    # print(UNIVERSE_SIZE + (NUM_OF_SUBSETS-subsetsIncluded))
    return UNIVERSE_SIZE + 2*(NUM_OF_SUBSETS-subsetsIncluded)

def getPopFitness(population: list, listOfSubsets: list) -> list:
    curGenFitness = []
    for state in population:
        stateFitness = fitnessValue(state,listOfSubsets)
        curGenFitness.append(stateFitness)
    return curGenFitness

def storeBestFitness(population: list, listOfSubsets: list, scpNum: int, genNum: int) -> list:
    for state in population:
        stateFitness = fitnessValue(state,listOfSubsets)
        fitnessValues[scpNum][genNum] = max(stateFitness,fitnessValues[scpNum][genNum])

def getNextGen(population: list, listOfSubsets: list) -> list:
    curGenFitness = np.array(getPopFitness(population,listOfSubsets))
    weights = np.array([])
    if (np.count_nonzero(curGenFitness) == 0):
        weights = np.full(POPULATION_SIZE,1/POPULATION_SIZE)
    elif(np.count_nonzero(curGenFitness) == 1):
        nonZeroIndex = -1
        for i in range(curGenFitness.shape[0]):
            if curGenFitness[i] != 0:
                nonZeroIndex = i
                break
        weights = np.full(POPULATION_SIZE,1/(POPULATION_SIZE+1))
        weights[nonZeroIndex] = 2/(POPULATION_SIZE+1)
    else:
        weights = np.full(POPULATION_SIZE,curGenFitness/curGenFitness.sum())
    
    nextGenPop = []
    for _ in range(POPULATION_SIZE):
        [firstParInd, secondParInd] = np.random.choice(np.arange(0,POPULATION_SIZE),size=2,replace=False,p=weights)
        child = reproduce(population[firstParInd], population[secondParInd])
        toMutate = random.random() < MUTATION_PROB
        if(toMutate):
            mutate(child)
        nextGenPop.append(child)
    # print(nextGenPop)
    return nextGenPop

    
def reproduce(parent1, parent2):
    crossOverPoint = random.randint(0,NUM_OF_SUBSETS-1)
    child = []
    for ind in range(crossOverPoint+1):
        child.append(parent1[ind])
    for ind in range(crossOverPoint+1, NUM_OF_SUBSETS):
        child.append(parent2[ind])
    return child

def mutate(state):
    index = random.randint(0,NUM_OF_SUBSETS-1)
    state[index] = 1 - state[index]
    
        
def main():
    scp = SetCoveringProblemCreator()

#    ********* You can use the following two functions in your program

    # subsets = scp.Create(usize=100,totalSets=200) # Creates a SetCoveringProblem with 200 subsets
    # print(len(subsets))
    # print()
    # listOfSubsets = scp.ReadSetsFromJson("scp_test.json") #Your submission program should read from scp_test.json file and provide a good solution for the SetCoveringProblem.
    # print(type(listOfSubsets))
    # print()
    
#    **********
#    Write your code for find the solution to the SCP given in scp_test.json file using Genetic algorithm.
    
    print("Roll no : 2021A7PS2066G")
    print(f"Number of subsets in scp_test.json file : {NUM_OF_SUBSETS}")
    # print(len(population))
    # print(fitnessValue([0,1,1,0,0], [[1,2,3,4,5,6], [1,2,3,7,8],[4,5,6,9,10],[7,8,9],[10]]))
    for scpNum in range(NUM_OF_RANDOM_SCP):
        subsets = scp.Create(100, NUM_OF_SUBSETS)
        scp.WriteSetsToJson(subsets,100,NUM_OF_SUBSETS)
        listOfSubsets = scp.ReadSetsFromJson(f"scp_{NUM_OF_SUBSETS}.json")
        population = generatePopulation(POPULATION_SIZE, listOfSubsets)
        # print(len(population))
        # print(np.array(population[0]).shape)
        # listOfSubsets = [[1,2,3,4,5,6], [1,2,3,7,8], [4,5,6,9,10], [7,8,9], [10]]
        # population = generatePopulation(POPULATION_SIZE,listOfSubsets)
        for genNum in range(NUM_OF_GENERATIONS):
            newPopulation = getNextGen(population,listOfSubsets)
            population = newPopulation
            storeBestFitness(population,listOfSubsets,scpNum,genNum)

    fitnessPlotValues = np.array(fitnessValues)
    y = np.mean(fitnessPlotValues, axis=0)
    x = np.arange(1,NUM_OF_GENERATIONS+1)
    plt.plot(x,y)
    plt.xlabel("Generation Number")
    plt.ylabel("Best Fitness")
    plt.title("Best fitness (avg over 10 SCP) and Generation Number")
    plt.show()
    # print(count)
    # print(population)
    # newPop = getNextGen(population,listOfSubsets)
    # population = newPop
    # newPop = getNextGen(population,listOfSubsets)





if __name__=='__main__':
    main()