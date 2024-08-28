from SetCoveringProblemCreator import *
import random
import numpy as np
import time


POPULATION_SIZE = 100
UNIVERSE_SIZE = 100
NUM_OF_SUBSETS = 50
NUM_OF_GENERATIONS = 3000
UNIVERSE = list(range(1,UNIVERSE_SIZE+1))
UNIVERSE_SET = set(UNIVERSE)
MUTATION_PROB = 0.3
CULLING_PARAM = 50
ELITISM_PARAM = 10


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
    subsetsIncluded = np.count_nonzero(np.array(state))
    curSet = set([])
    for i in range(n):
        if state[i] == 1:
            for num in listOfSubsets[i]:
                curSet.add(num)
    if(curSet != UNIVERSE_SET):
        return 5 + len(curSet)/6
    return UNIVERSE_SIZE + 2*(NUM_OF_SUBSETS-subsetsIncluded)

def getPopFitness(population: list, listOfSubsets: list) -> list:
    curGenFitness = []
    for state in population:
        stateFitness = fitnessValue(state,listOfSubsets)
        curGenFitness.append(stateFitness)
    return curGenFitness


def getNextGen(population: list, listOfSubsets: list, cullingParam: int) -> list:
    curGenFitness = getPopFitness(population,listOfSubsets)
    weights = np.array(curGenFitness)/curGenFitness.sum()
    
    nextGenPop = []
    for _ in range(POPULATION_SIZE + cullingParam):
        [firstParInd, secondParInd] = np.random.choice(np.arange(0,POPULATION_SIZE),size=2,replace=False,p=weights)
        child = reproduce(population[firstParInd], population[secondParInd])
        toMutate = random.random() < MUTATION_PROB
        if(toMutate):
            mutate(child)
        nextGenPop.append(child)
    nextGenFitness = getPopFitness(nextGenPop,listOfSubsets)
    sortedNextGen = [x for _,x in sorted(zip(nextGenFitness,nextGenPop), reverse=True)]
    sortedNextGen = sortedNextGen[:POPULATION_SIZE]
    return sortedNextGen

def getBestState(population: list, listOfSubsets: list) -> list:
    maxFitnessVal = -1e9
    bestState = []
    for state in population:
        stateFitness = fitnessValue(state, listOfSubsets)
        if(stateFitness > maxFitnessVal):
            maxFitnessVal = stateFitness
            bestState = state
    return bestState

def getNumberOfSubsetsInState(state: list) -> int:
    subsetsIncluded = 0
    for bit in state:
        if bit == 1: subsetsIncluded += 1
    return subsetsIncluded
    
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

def printSolution(totalTime: int, bestState: list, bestFitness: int, bestStateNumOfSubsets: int):
    print("Solution :")
    ind = 0
    for bit in bestState:
        end = ", "
        if (ind == len(bestState)-1) :
            end = ""
        print(f"{ind}:{bit}", end=end)
        ind += 1
    print("")
    print(f"Fitness value of best state : {bestFitness}")
    print(f"Minimum number of subsets that can cover the Universe-set : {bestStateNumOfSubsets}")
    print(f"Time taken : {totalTime} seconds")
    
        
def main():
    start_time = time.time()
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
    start_time = time.time()
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    global NUM_OF_SUBSETS
    NUM_OF_SUBSETS = len(listOfSubsets)

    print("Roll no : 2021A7PS2066G")
    print(f"Number of subsets in scp_test.json file : {NUM_OF_SUBSETS}")
    bestState = []
    bestFitness = -1e9
    bestStateNumOfSubsets = NUM_OF_SUBSETS
    population = generatePopulation(POPULATION_SIZE,listOfSubsets)

    for _ in range(NUM_OF_GENERATIONS):
        newPopulation = getNextGen(population,listOfSubsets,CULLING_PARAM)
        newGenBestState = getBestState(newPopulation,listOfSubsets)
        newGenBestFitness = fitnessValue(newGenBestState,listOfSubsets)
        if(newGenBestFitness > bestFitness):
            bestFitness = newGenBestFitness
            bestState = newGenBestState
            bestStateNumOfSubsets = getNumberOfSubsetsInState(bestState)
        population = newPopulation
        elapsedTime = time.time() - start_time
        if(elapsedTime >= 40): 
            break 
    
    end_time = time.time()
    totalTime = end_time - start_time
    printSolution(totalTime, bestState, bestFitness,bestStateNumOfSubsets)



if __name__=='__main__':
    main()