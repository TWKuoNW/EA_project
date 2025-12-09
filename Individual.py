#
# Individual.py
#
#

import math

#Base class for all individual types
#
class Individual:
    """
    Individual
    """
    minMutRate=1e-100
    maxMutRate=1
    learningRate=None
    uniprng=None
    normprng=None
    ObjFunc=None

    def __init__(self):
        self.objectives=self.__class__.ObjFunc(self.state)
        self.mutRate=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
        self.numObj=len(self.objectives)
        self.frontRank=None
        self.crowdDist=None

    def mutateMutRate(self):
        self.mutRate=self.mutRate*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.mutRate < self.minMutRate: 
            self.mutRate=self.minMutRate
        if self.mutRate > self.maxMutRate: 
            self.mutRate=self.maxMutRate

    def evaluateObjectives(self):
        if self.objectives == None: 
            self.objectives=self.__class__.ObjFunc(self.state)

    def dominates(self,other):
        dominatesCount=0
        equalsCount=0
        inferiorCount=0

        i=0
        while i < self.numObj:
            ### minimize mana cost
            if i==0:
                if self.objectives[i] < other.objectives[i]: dominatesCount+=1
                elif self.objectives[i] > other.objectives[i]: inferiorCount+=1
                else: equalsCount+=1
                i+=1

            ### maximize spell damage
            else:
                if self.objectives[i] > other.objectives[i]: dominatesCount+=1
                elif self.objectives[i] < other.objectives[i]: inferiorCount+=1
                else: equalsCount+=1
                i+=1

        if equalsCount == self.numObj: return 0 # the two are non-dominating
        elif dominatesCount+equalsCount == self.numObj: return 1 # self dominates other
        elif inferiorCount+equalsCount == self.numObj: return -1 # other dominates self
        else: return 0 # the two are non-dominating

    def compareRankAndCrowding(self, other):
        if self is other: return 0

        if self.frontRank < other.frontRank: return 1
        elif self.frontRank > other.frontRank: return -1
        else:
            if self.crowdDist > other.crowdDist: return 1
            elif self.crowdDist < other.crowdDist: return -1
            else: return 0

    def distance(self, other, normalizationVec=[None]):
        """
        Compute distance between self & other in objective space
        """
        # check if self vs self
        if self is other:
            return 0.0

        #set default normalization to 1.0, if not specified
        if normalizationVec[0] == None:
            normalizationVec=[1.0]*self.numObj

        # compute normalized Euclidian distance
        distance=0
        i=0
        while i < self.numObj:
            tmp=(self.objectives[i]-other.objectives[i])/normalizationVec[i]
            distance+=(tmp*tmp)
            i+=1

        distance=math.sqrt(distance)

        return distance

class AgentIndividual(Individual):
    """
    ActorIndividual
    """
    nGenes=None
    geneRange=None
    numTryPerMut = None
    blockActionSize = None

    def __init__(self):
        self.state = []
        init_genes = 2 * self.__class__.blockActionSize 
        for _ in range(init_genes):
            self.state.append(self.uniprng.uniform(0.0, float(self.geneRange)))
        super().__init__()

    def crossover(self, other):
        #perform crossover "in-place"
        for i in range(self.nGenes):
            if self.uniprng.random() < 0.5:
                tmp=self.state[i]
                self.state[i]=other.state[i]
                other.state[i]=tmp

        self.objectives=None
        other.objectives=None

    def mutate(self):
        """
        Simplified mutation:
        - Do NOT run internal evolution / selection.
        - Just append one new action block (a sequence of actions) to the end.
        - The outer EA (selection) will decide whether this longer sequence is good.
        """

        # Optionally: adapt mutation rate (you can also remove this line if not needed)
        self.mutateMutRate()

        # If already at maximum allowed length, do nothing
        if len(self.state) >= self.__class__.nGenes:
            return

        # One block = blockActionSize actions, each action uses 2 genes (x, y)
        genes_per_block = 2 * self.__class__.blockActionSize

        # Avoid exceeding nGenes
        max_addable = self.__class__.nGenes - len(self.state)
        genes_to_add = min(genes_per_block, max_addable)

        # Append genes_to_add genes (in pairs: [a, b])
        for _ in range(genes_to_add // 2):
            a = self.uniprng.uniform(0.0, float(self.geneRange))
            b = self.uniprng.uniform(0.0, float(self.geneRange))
            self.state.extend([a, b])

        # Let the outer EA re-evaluate this individual later
        self.objectives = None


    
    def __str__(self):
        return (
            str(self.state) + '\t' +
            '%0.8e' % self.mutRate + '\t' +
            'nSeqSteps: ' + str(self.objectives[0]) + '\t' +
            'rewardEnd: ' + str(self.objectives[1]) + '\t' +
            'crowdDist: ' + '%0.8e' % self.crowdDist + ' ' +
            'frontRank: ' + str(self.frontRank)
        )
