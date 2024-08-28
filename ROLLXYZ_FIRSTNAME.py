from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import random
import numpy as np
import time

NUM_OF_RANDOM_SCP = 10
POPULATION_SIZE = 100
UNIVERSE_SIZE = 100
NUM_OF_SUBSETS = 50
NUM_OF_GENERATIONS = 400
UNIVERSE = list(range(1,UNIVERSE_SIZE+1))
UNIVERSE_SET = set(UNIVERSE)
MUTATION_PROB = 0.3
CULLING_PARAM = 0
ELITISM_PARAM = 10
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
        return 5 + len(curSet)/6
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
    # if genNum > 0:
    #     fitnessValues[scpNum][genNum] = max(fitnessValues[scpNum][genNum], fitnessValues[scpNum][genNum-1])

def getNextGen(population: list, listOfSubsets: list, cullingParam: int, elitismParam: int) -> list:
    curGenFitness = np.array(getPopFitness(population,listOfSubsets))
    sortedCurGen = [x for _,x in sorted(zip(getPopFitness(population,listOfSubsets),population), reverse=True)]
    eliteParents = sortedCurGen[:elitismParam]
    # if (np.count_nonzero(curGenFitness) == 0):
    #     weights = np.full(POPULATION_SIZE,1/POPULATION_SIZE)
    # elif(np.count_nonzero(curGenFitness) == 1):
    #     nonZeroIndex = -1
    #     for i in range(curGenFitness.shape[0]):
    #         if curGenFitness[i] != 0:
    #             nonZeroIndex = i
    #             break
    #     weights = np.full(POPULATION_SIZE,1/(POPULATION_SIZE+1))
    #     weights[nonZeroIndex] = 2/(POPULATION_SIZE+1)
    # else:
    weights = curGenFitness/curGenFitness.sum()
    
    nextGenPop = []
    for _ in range(POPULATION_SIZE + cullingParam):
        [firstParInd, secondParInd] = np.random.choice(np.arange(0,POPULATION_SIZE),size=2,replace=False,p=weights)
        child = reproduce(population[firstParInd], population[secondParInd])
        toMutate = random.random() < MUTATION_PROB
        if(toMutate):
            mutate(child)
        nextGenPop.append(child)
    # print(nextGenPop)
    nextGenFitness = getPopFitness(nextGenPop,listOfSubsets)
    sortedNextGen = [x for _,x in sorted(zip(nextGenFitness,nextGenPop), reverse=True)]
    sortedNextGen = sortedNextGen[:POPULATION_SIZE]
    for ind in range(POPULATION_SIZE-elitismParam,POPULATION_SIZE):
        sortedNextGen[ind] = eliteParents[POPULATION_SIZE-ind-1]

    return sortedNextGen

    
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
    start_time = time.time()
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
            newPopulation = getNextGen(population,listOfSubsets,CULLING_PARAM, ELITISM_PARAM)
            population = newPopulation
            storeBestFitness(population,listOfSubsets,scpNum,genNum)
    
    end_time = time.time()

    fitnessPlotValues = np.array(fitnessValues)
    y = np.mean(fitnessPlotValues, axis=0)
    x = np.arange(1,NUM_OF_GENERATIONS+1)
    plt.plot(x,y)
    plt.xlabel("Generation Number",fontsize=14)
    plt.ylabel("Best Fitness",fontsize=14)
    plt.title(f"Best fitness (avg over 10 SCP) and Generation Number for |S|={NUM_OF_SUBSETS} and K={CULLING_PARAM}",fontsize=15)
    plt.show()

    y1 = np.std(fitnessPlotValues,axis=0)
    plt.plot(x,y1)
    plt.xlabel("Generation Number",fontsize=14)
    plt.ylabel("Standard Deviation",fontsize=14)
    plt.title(f"Std deviation (over 10 random SCP) and Generation Number for |S|={NUM_OF_SUBSETS} and K={CULLING_PARAM}",fontsize=15)
    plt.show()
    # print(count)
    # print(population)
    # newPop = getNextGen(population,listOfSubsets)
    # population = newPop
    # newPop = getNextGen(population,listOfSubsets)
    finalAvgBestFitness = y[NUM_OF_GENERATIONS-1]
    finalStdBestFitness = y1[NUM_OF_GENERATIONS-1]
    with open("Records.txt", "a") as file:
        file.write(f"Number of subsets: {NUM_OF_SUBSETS}, Final best fitness: {finalAvgBestFitness}, Final std dev: {finalStdBestFitness}\n")

    print(f"Time elapsed: {end_time-start_time}")



if __name__=='__main__':
    main()