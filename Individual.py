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
        Mutation logic:
        1. First record the current performance of this action sequence in the environment.
        2. Run multiple experiments:
        - Copy the current action sequence
        - Append a small new action block at the end
        - Run it in the environment to see whether the reward improves
        3. If any experiment produces a better result, adopt the best candidate.
        Otherwise, keep the original sequence unchanged.
        """

        # 1. Make sure we have the current objective values
        if self.objectives is None:
            self.objectives = self.__class__.ObjFunc(self.state)

        # Current step count and reward
        base_steps, base_reward = self.objectives

        best_state = self.state[:]          # The best gene sequence so far
        best_steps = base_steps
        best_reward = base_reward

        # If maximum allowed length is reached, do not add more blocks
        # nGenes = "maximum allowed number of genes"
        if len(self.state) >= self.__class__.nGenes:
            return

        # 2. Try multiple candidate mutations (numTryPerMut times)
        for _ in range(self.__class__.numTryPerMut):
            # Copy the current individual's state
            candidate = self.state[:]

            # How many genes to add this time:
            # blockActionSize actions, each action = 2 values
            genes_to_add = 2 * self.__class__.blockActionSize

            # Skip if adding this block would exceed maximum length
            if len(candidate) + genes_to_add > self.__class__.nGenes:
                continue

            # Append one action block consisting of blockActionSize actions
            for _ in range(self.__class__.blockActionSize):
                a = self.uniprng.uniform(0.0, float(self.geneRange))
                b = self.uniprng.uniform(0.0, float(self.geneRange))
                candidate.extend([a, b])

            # Evaluate this candidate in the environment
            cand_steps, cand_reward = self.__class__.ObjFunc(candidate)

            # Compare which one is better:
            #   1. Higher reward → better
            #   2. If reward is equal, fewer steps → better
            is_better = False
            if cand_reward > best_reward:
                is_better = True
            elif cand_reward == best_reward and cand_steps < best_steps:
                is_better = True

            if is_better:
                best_state = candidate
                best_steps = cand_steps
                best_reward = cand_reward

        # 3. Update the individual if the best candidate is better than the original
        #    (Compare again with base values to ensure actual improvement)
        if (best_reward > base_reward) or (best_reward == base_reward and best_steps < base_steps):
            self.state = best_state
            self.objectives = [best_steps, best_reward]
        else:
            # No improvement → keep the original (objectives remain base)
            self.objectives = [base_steps, base_reward]


    
    def __str__(self):
        return (
            str(self.state) + '\t' +
            '%0.8e' % self.mutRate + '\t' +
            'nSeqSteps: ' + str(self.objectives[0]) + '\t' +
            'rewardEnd: ' + str(self.objectives[1]) + '\t' +
            'crowdDist: ' + '%0.8e' % self.crowdDist + ' ' +
            'frontRank: ' + str(self.frontRank)
        )
