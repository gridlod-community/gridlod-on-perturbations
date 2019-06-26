import numpy as np
import gridlod
import scipy.sparse

# Pick a discretization of the domain
NCoarseElement = np.array([8, 8])
NWorldCoarse = np.array([32, 32])

world = gridlod.world.World(NWorldCoarse, NCoarseElement)

# Pick element (1,1)
TInd = gridlod.util.convertpCoordIndexToLinearIndex(NWorldCoarse-1, np.array([1, 1]))

# Make an _element patch_ around it. This is required, since IH is defined
# in terms of the neighboring elements.
patch = gridlod.world.Patch(world, 1, TInd)

# Make an indicator function for the center element in the patch
indicatorT = np.zeros(patch.NtFine, dtype=np.float64)
tIndicesForT = gridlod.util.extractElementFine(patch.NPatchCoarse,
                                               NCoarseElement,
                                               patch.iElementPatchCoarse,
                                               extractElements=True)
indicatorT[tIndicesForT] = 1

# Set coefficient A in patch
a = 1 + 0*np.random.rand(patch.NtFine)

# Compute IH for the patch
IH = gridlod.interp.L2ProjectionPatchMatrix(patch)
# Compute AFine for the patch
AFine = gridlod.fem.assemblePatchMatrix(patch.NPatchFine, world.ALocFine, a)
# Compute MFine for only the center element
MFine = gridlod.fem.assemblePatchMatrix(patch.NPatchFine, world.MLocFine, indicatorT)
# Identity matrix
E = scipy.sparse.identity(IH.shape[1])

# Prolongation
P = gridlod.fem.assembleProlongationMatrix(patch.NPatchCoarse, world.NCoarseElement)

# Setup left and right hand sides of the generalized eigenvalue problem
L = (E - P*IH).T*MFine*(E-P*IH)
R = AFine

# Compute the supremum
supremum = scipy.sparse.linalg.eigsh(L[1:,1:], 1, R[1:,1:])[0]
print(supremum)
print(np.sqrt(supremum))
print(NWorldCoarse[0]*np.sqrt(supremum))
