import numpy as np
import time


def Sphere(x):
    return np.sum(x ** 2)


def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def FHO(Pop, CostFunction, VarMin, VarMax, MaxFes):
    nPop, VarNumber = Pop.shape
    Iter = 0
    FEs = 0

    # Initialization
    # Pop = np.random.uniform(VarMin, VarMax, (nPop, VarNumber))
    Cost = np.array([CostFunction(x) for x in Pop])
    FEs += nPop

    # Sort Population
    SortOrder = np.argsort(Cost)
    Pop = Pop[SortOrder, :]
    BestPop = Pop[0, :]
    GB = Cost[0]
    SP = np.mean(Pop, axis=0)

    # Fire Hawks
    HN = np.random.randint(1, np.ceil(nPop / 5) + 1)
    FHPops = Pop[:HN, :]

    # Prey
    Pop2 = Pop[HN:, :]
    ct = time.time()
    # Main Loop
    while FEs < MaxFes:
        Iter += 1
        PopTot = []

        for i in range(len(Pop2)):
            PR = Pop2[i, :]
            FHl = FHPops[i % HN, :]
            SPl = np.mean(PR)

            Ir = np.random.uniform(0, 1, size=(2,))
            FHnear = FHPops[np.random.randint(HN), :]
            FHl_new = FHl + Ir[0] * (GB - FHl) - Ir[1] * (FHnear - FHl)
            FHl_new = np.maximum(FHl_new, VarMin)
            FHl_new = np.minimum(FHl_new, VarMax)
            PopTot.append(FHl_new)

            for _ in range(2 * len(PR)):
                Ir = np.random.uniform(0, 1, size=(2,))
                PRq_new1 = PR + Ir[0] * (FHl - SPl)
                PRq_new1 = np.maximum(PRq_new1, VarMin)
                PRq_new1 = np.minimum(PRq_new1, VarMax)
                PopTot.append(PRq_new1)

                FHAlter = FHPops[np.random.randint(HN), :]
                PRq_new2 = PR + Ir[0] * (FHAlter - SP)
                PRq_new2 = np.maximum(PRq_new2, VarMin)
                PRq_new2 = np.minimum(PRq_new2, VarMax)
                PopTot.append(PRq_new2)

        PopTot = np.array(PopTot)
        Cost = np.array([CostFunction(x) for x in PopTot])
        FEs += len(PopTot)

        # Sort Population
        SortOrder = np.argsort(Cost)
        PopTot = PopTot[SortOrder, :]
        Pop = PopTot[:nPop, :]
        HN = np.random.randint(1, np.ceil(nPop / 5) + 1)
        BestPop = Pop[0, :]
        GB = min(GB, np.min(Cost))
        FHPops = Pop[:HN, :]
        Pop2 = Pop[HN:, :]

        # Update Bests
        if Cost[0,0] < GB:
            BestPos = BestPop

        # Store Best Cost Ever Found
        if 'BestCosts' not in locals():
            BestCosts = np.zeros((MaxFes,))
        BestCosts[Iter - 1] = GB

    Destination_fitness = GB
    Destination_position = BestPop[0, 0]
    Convergence_curve = BestCosts
    ct = time.time() - ct
    return Destination_fitness, Convergence_curve, Destination_position, ct
