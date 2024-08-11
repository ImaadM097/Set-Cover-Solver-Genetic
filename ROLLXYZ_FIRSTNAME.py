from SetCoveringProblemCreator import *
import matplotlib
import random

NUM_OF_RANDOM_SCP = 10
POPULATION_SIZE = 50
UNIVERSE_SIZE = 100
NUM_OF_SUBSETS = 50
NUM_OF_GENERATIONS = 50
UNIVERSE = list(range(1,UNIVERSE_SIZE+1))
UNIVERSE_SET = set(range(1,UNIVERSE_SIZE))

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
    # listOfSubsets = [[1,2,3,4,5,6], [1,2,3,7,8], [4,5,6,9,10], [7,8,9], [10]]
    # generatePopulation(3,listOfSubsets)
    print("Roll no : 2021A7PS2066G")
    fitnessValues = [[0 for _ in range(NUM_OF_GENERATIONS)] for _ in range(NUM_OF_RANDOM_SCP)]
    subsets = scp.Create(UNIVERSE_SIZE, NUM_OF_SUBSETS)
    print(f"Number of subsets in scp_test.json file : {len(subsets)}")
    listOfSubsets = scp.ReadSetsFromJson("scp_test.json")
    population = generatePopulation(POPULATION_SIZE, listOfSubsets)
    print(population)





if __name__=='__main__':
    main()