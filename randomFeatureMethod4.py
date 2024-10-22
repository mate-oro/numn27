import numpy as np
from scipy.optimize import  least_squares
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from numdifftools import Derivative, Gradient
from timerDeco import timeit
import matplotlib.pyplot as plt
import line_profiler

def main():
    rfm1 = rfm1d(5, 50, 20, 5)
    print("Initialization done.")

    rfm1.optimize()

class rfm1d(object):
    @line_profiler.profile
    def __init__(self, numParts: int, numFuns: int, numCollPoints: int, randRange: float, featFunType: str = "sin", partType: str = "B", h: float = 1e-6, rescaling: bool = False) -> None:
        """
        Initialzing the RFM so that we can pass it to the solver

        Args:
            numParts (int): Number of partitions in the PoU
            numFuns (int): Number of random feature functions per partition
            numCollPoints (int): Number of collocation points that are used to deal with the PDE and Boundary Condition
            randRange (float): R value used for the featur function values that are drawn from Unif(-R,R) distribution
        """
        self.numParts = numParts
        self.numFuns = numFuns
        self.radius = 4/numParts
        self.numCollPoints = numCollPoints
        self.randRange = randRange
        self.featFunType = featFunType.lower()
        self.lossWeightI = np.ones((numCollPoints - 2))
        self.lossWeightB = 1 * np.ones((2)) # 50, 50, 25, 25
        self.lossWeightE1 = 1 * np.ones((numParts - 1))
        self.lossWeightE2 = 1 * np.ones((numParts - 1))
        self.h = h
        self.lamb = 25
        self.partType = partType.lower()

        # Values for the feature functions, where each list item corresponds to one partition
        self.featVals = [np.random.uniform(-1, 1, (numFuns, 2)) * randRange for _ in range(numParts)]

        # Coefficients for the feature functions, where each row corresponds to one partition
        self.featCoeffs = np.random.randn(numParts, numFuns)

        # List of the coordinates of the centerpoints of the partitions
        self.partPoints = [4 * (2 * i - 1) / numParts for i in range(1, numParts + 1)]

        # List of the coordinates of the collocation points
        self.collPoints = np.linspace(0, 8, numCollPoints)

        # Calculating the values of the Partition indicator functions at the collocation points -- in nore: Ψ 
        self.indicatorVals: np.ndarray = np.zeros((numParts, numCollPoints))
        for i in range(numParts):
            self.indicatorVals[i] = self.partFun(self.collPoints, i)

        if rescaling:
            print("Started rescaling")
            self.lossWeightI, self.lossWeightB = self.weightRescale(10)

        # Jac sparsity
        print("Started building the Jacobian")
        if self.partType == "a":
            s = 1
            J: lil_matrix = lil_matrix((numCollPoints + 2 * (self.numParts - 1), self.featCoeffs.size))
            C: lil_matrix = lil_matrix((numCollPoints + 2 * (self.numParts - 1), self.featCoeffs.size), dtype=int)
        else:
            s = 4/5
            J: lil_matrix = lil_matrix((numCollPoints, self.featCoeffs.size))
            C: lil_matrix = lil_matrix((numCollPoints, self.featCoeffs.size), dtype=int)
        for i in range(1, numCollPoints-1):
            for j in range(numParts):
                if abs(self.collPoints[i] - self.partPoints[j]) * s <= (self.radius + self.h):
                    C[i+1, j*numFuns : (j+1)*numFuns] = 1
                    J[i+1, j*numFuns : (j+1)*numFuns] = [self.gradUnj(self.collPoints[i], j, k) for k in range(numFuns)]
        C[0, :numFuns] = 1
        C[1, (numParts-1)*numFuns:] = 1
        match self.featFunType:
            case "tanh":
                featFun = lambda x, n, j: np.tanh(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
                featFunD = lambda x, n, j: (self.featVals[n][j, 0] / self.radius) / np.pow(np.cosh(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1]), 2)
            case "cos":
                featFun = lambda x, n, j: np.cos(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
                featFunD = lambda x, n, j: -np.sin(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1]) * self.featVals[n][j, 0] / self.radius
            case _:
                featFun = lambda x, n, j: np.sin(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
                featFunD = lambda x, n, j: np.cos(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1]) * self.featVals[n][j, 0] / self.radius
                
        J[0, :numFuns] =              [self.lossWeightB[ 0] * self.indicatorVals[ 0, 0] * featFun(self.collPoints[ 0],  0, k) for k in range(numFuns)]
        J[1, (numParts-1)*numFuns:] = [self.lossWeightB[-1] * self.indicatorVals[-1,-1] * featFun(self.collPoints[-1], -1, k) for k in range(numFuns)]
        if self.partType == "a":
            edges = np.linspace(0, 8, self.numParts + 1)[1:-1]
            for e in range(numParts - 1):
                J[self.numCollPoints + e, e*numFuns     : (e+1)*numFuns] = [-self.lossWeightE1[e] * (featFun(edges[e], e,   k)) for k in range(numFuns)]
                J[self.numCollPoints + e, (e+1)*numFuns : (e+2)*numFuns] = [self.lossWeightE1[e]  * (featFun(edges[e], e+1, k)) for k in range(numFuns)]
                C[self.numCollPoints + e, e*numFuns : (e+2)*numFuns] = 1
                J[self.numCollPoints + numParts - 1 + e, e*numFuns     : (e+1)*numFuns] = [-self.lossWeightE2[e] * featFunD(edges[e], e,   k) for k in range(numFuns)]
                J[self.numCollPoints + numParts - 1 + e, (e+1)*numFuns : (e+2)*numFuns] = [self.lossWeightE2[e]  * featFunD(edges[e], e+1, k) for k in range(numFuns)]
                C[self.numCollPoints + numParts - 1 + e, e*numFuns : (e+2)*numFuns] = 1
        self.jacSparsity: lil_matrix = C
        self.jac = J.toarray()
    
    def weightRescale(self, c: float = 10) -> tuple[np.ndarray, np.ndarray]:
        inner = np.zeros((self.numCollPoints, self.numParts, self.numFuns))
        bounday = np.zeros((2, self.numParts, self.numFuns))
        b = np.array([0., 8.])
        if self.featFunType == "cos":
            featFun = lambda x, n, j: np.cos(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
        elif self.featFunType == "tanh":
            featFun = lambda x, n, j: np.tanh(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
        else:
            featFun = lambda x, n, j: np.sin(self.featVals[n][j, 0] * self.renorm(x, self.partPoints[n], self.radius) + self.featVals[n][j, 1])
        for i in range(self.numParts):
            for j in range(self.numFuns):
                fun = lambda x: featFun(x, i, j) * self.partFun(x, i)
                L = Derivative(fun, self.h, method = "complex", n = 2)
                Lx = L(self.collPoints)
                inner[:, i, j] = np.abs(Lx - self.lamb * fun(self.collPoints))
                bounday[:, i, j] = np.abs(fun(b))
        return c / inner.max(axis = (1, 2)), c / bounday.max(axis = (1, 2))

    def gradUnj(self, x: float, n: int,  j: int) -> float:
        """The derivative second derivative wrt x of the derivative of the sol wrt u_n,j.

        Args:
            x (float): point at which the second derivative is going to be taken
            n (int): index of partition
            j (int): index of function in partition

        Returns:
            Derivative: the value of (d/dx)^2 (d/du_n,j) sol(x, u)
        """
        feat = self.featVals[n]
        match self.featFunType:
            case "tanh":
                deriv =  lambda y: self.partFun(y, n) * np.tanh(feat[j, 0] * self.renorm(y, self.partPoints[n], self.radius) + feat[j, 1])
            case "cos":
                deriv =  lambda y: self.partFun(y, n) * np.cos(feat[j, 0]  * self.renorm(y, self.partPoints[n], self.radius) + feat[j, 1])
            case _:
                deriv =  lambda y: self.partFun(y, n) * np.sin(feat[j, 0]  * self.renorm(y, self.partPoints[n], self.radius) + feat[j, 1])
        D = Derivative(deriv, self.h, method = "complex", n = 2) 
        return D(x) - self.lamb * deriv(x)  # type: ignore
    
    def J(self, x):
        return self.jac

    def partFun(self, x: float|np.ndarray, n: int) -> float|np.ndarray:
        """
        Calculates the partition indicator function with a smooth boundray

        Args:
            x (float): input of the functiion

        Returns:
            float: output of the function
        """
        y = self.renorm(x, self.partPoints[n], self.radius)
        match self.partType:
            case "a":
                if type(y) == float or type(y) == np.float64 or type(y) == np.complex128:
                    if (-1 <= y < 1) and n < self.numParts -1:
                        return float(1)
                    elif (-1 <= y <= 1) and n == self.numParts -1:
                        return float(1)
                    else:
                        return float(0)
                elif type(y) == np.ndarray:
                    # Create a boolean mask for the condition
                    if n < self.numParts -1:
                        mask = ((y >= -1) & (y < 1))
                    elif n == self.numParts -1:
                        mask = ((y >= -1) & (y <= 1))
                    

                    p = np.zeros_like(y)
                    p[mask] = 1
                    return p
                raise Exception(f"Invalid input type: {type(y)}")
            case _:
                if type(y) == float or type(y) == np.float64 or type(y) == np.complex128:
                    if n == 0:
                        if -1 <= y < 3/4:
                            return float(1)
                        elif 3/4 <= y < 5/4:
                            return (1 - np.sin(2 * np.pi * y)) / 2
                        else:
                            return float(0)
                    elif n == self.numParts - 1:
                        if -5/4 <= y < -3/4:
                            return (1 + np.sin(2 * np.pi * y)) / 2
                        elif -3/4 <= y <= 1:
                            return float(1)
                        else:
                            return float(0)
                    else:
                        if -5/4 <= y < -3/4:
                            return (1 + np.sin(2 * np.pi * y)) / 2
                        elif -3/4 <= y < 3/4:
                            return float(1)
                        elif 3/4 <= y < 5/4:
                            return (1 - np.sin(2 * np.pi * y)) / 2
                        else:
                            return float(0)
                elif type(y) == np.ndarray:
                    # Create a boolean mask for the condition
                    if n == 0:
                        mask1 = False
                        mask2 = (y <= 3/4) & (y <= 1)
                        mask3 = (y > 3/4) & (y < 5/4)
                    elif n == self.numParts - 1:
                        mask1 = (y > -5/4) & (y < -3/4)
                        mask2 = (y >= -3/4) & (y >= -1)
                        mask3 = False
                    else:
                        mask1 = (y > -5/4) & (y < -3/4)
                        mask2 = (y >= -3/4) & (y <= 3/4)
                        mask3 = (y > 3/4) & (y < 5/4)

                    p = np.zeros_like(x)
                    p[mask1] = (1 + np.sin(2 * np.pi * y[mask1])) / 2 # type: ignore
                    p[mask2] = 1
                    p[mask3] = (1 - np.sin(2 * np.pi * y[mask3])) / 2 # type: ignore
                    return p
                raise Exception(f"Invalid input type: {type(x)}")

    def renorm(self, x: float|np.ndarray, center: float, scale: float) -> float|np.ndarray:
        """
        Calculating the renormalised coordinate

        Args:
            x (float): coordinate of point
            center (float): center for the renormalised coordinate
            scale (float): scale for zhe renormalised coordinate

        Returns:
            float: the point in renormalised the coordinate
        """
        return (x - center) / scale

    def evalSol(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the approx solution at a point 'x'

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution

        Returns:
            float: the value of the approximation
        """
        match self.featFunType:
            case "tanh":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = np.tanh(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n)
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = np.tanh(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))))
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case "cos":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = np.cos(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n)
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = np.cos(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))))
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case _:
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = np.sin(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n)
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = np.sin(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))))
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")

    def evalDDA(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the Second Derivative of the approx solution at a point 'x', with case "a"

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution

        Returns:
            float: the value of the approximation
        """
        match self.featFunType:
            case "tanh":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = -2 * np.sinh(feats[:, 0] * y + feats[:, 1]) * np.pow(np.cosh(feats[:, 0] * y + feats[:, 1]), 2) * np.pow(feats[:, 0], 2) / np.pow(np.cosh(feats[:, 0] * y + feats[:, 1]), 3) / self.radius**2
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = -2 * np.sinh(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size)))) * np.pow(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))), 2) * np.outer(np.pow(feats[:, 0], 2), np.ones((x.size))) / np.pow(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))), 3) / self.radius**2
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case "cos":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = -np.cos(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n) * feats[:, 0] * feats[:, 0] / self.radius**2
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = -np.cos(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size)))) * np.outer(np.pow(feats[:, 0], 2), np.ones((x.size))) / self.radius**2
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case _:
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = -np.sin(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n) * feats[:, 0] * feats[:, 0] / self.radius**2
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = -np.sin(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size)))) * np.outer(np.pow(feats[:, 0], 2), np.ones((x.size))) / self.radius**2
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
    
    def evalDA(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the First Derivative of the approx solution at a point 'x', with case "a"

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution

        Returns:
            float: the value of the approximation
        """
        match self.featFunType:
            case "tanh":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = feats[:, 0] / np.pow(np.cosh(feats[:, 0] * y + feats[:, 1]), 2) / self.radius
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = np.outer(feats[:, 0], np.ones((x.size))) / np.pow(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size))), 2) / self.radius
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case "cos":
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = -np.sin(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n) * feats[:, 0] / self.radius
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = -np.sin(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size)))) * np.outer(feats[:, 0], np.ones((x.size))) / self.radius
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")
            case _:
                if type(x) == float:
                    u: float = 0 # type: ignore
                    for (feats, center, coeffs, n) in zip(self.featVals, self.partPoints, self.featCoeffs, range(self.numParts)):
                        y = float(self.renorm(x, center, self.radius))
                        un = np.cos(feats[:, 0] * y + feats[:, 1])  * self.partFun(x, n) * feats[:, 0] / self.radius
                        u += np.dot(coeffs, un)
                    return u
                elif type(x) == np.ndarray:
                    u: np.ndarray = np.zeros_like(x)
                    for j in range(self.numParts):
                        y = self.renorm(x, self.partPoints[j], self.radius)
                        feats = self.featVals[j]
                        un = np.cos(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((x.size)))) * np.outer(feats[:, 0], np.ones((x.size))) / self.radius
                        u += (self.featCoeffs[j] @ un) * self.partFun(x, j)
                    return u
                raise Exception(f"Invalid input type: {type(x)}")

    def evalSolAPart(self, x: float | np.ndarray, n: int) -> float | np.ndarray:
        """
        Evaluate the value of the approx solution at a point 'x', with case "a", on partition n

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution
            n (int): index of partition

        Returns:
            float: the value of the approximation
        """
        match self.featFunType:
            case "tanh":
                featFun = lambda x: np.tanh(x)
            case "cos":
                featFun = lambda x: np.cos(x)
            case _:
                featFun = lambda x: np.sin(x)
        if type(x) == float:
            x = x * np.ones((1))
        y = self.renorm(x, self.partPoints[n], self.radius)
        feats = self.featVals[n]
        un = featFun(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones_like(x)))
        return np.dot(self.featCoeffs[n], un)

    def evalDAPart(self, x: float, n: int) -> float:
        """
        Evaluate the First Derivative of the approx solution at a point 'x', with case "a", on partition n

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution

        Returns:
            float: the value of the approximation
        """
        match self.featFunType:
            case "tanh":
                featFunD = lambda x: np.pow(np.cosh(x), -2)
            case "cos":
                featFunD = lambda x: -np.sin(x)
            case _:
                featFunD = lambda x: np.cos(x)
        y = self.renorm(x, self.partPoints[n], self.radius)
        feats = self.featVals[n]
        un = featFunD(feats[:, 0] * y + feats[:, 1]) * feats[:, 0] / self.radius
        return np.inner(self.featCoeffs[n], un)

    @line_profiler.profile
    def evalCollocs(self) -> np.ndarray:
        """
        Evaluate the approx solution at the collocation points

        Args:
            x (float): the point at whitch we want to evaluate the u approximate solution

        Returns:
            float: the value of the approximation
        """
        u: np.ndarray = np.zeros_like(self.collPoints)
        match self.featFunType:
            case "tanh":
                featFun = lambda x: np.tanh(x)
            case "cos":
                featFun = lambda x: np.cos(x)
            case _:
                featFun = lambda x: np.sin(x)
        for j in range(self.numParts):
            y = self.renorm(self.collPoints, self.partPoints[j], self.radius)
            feats = self.featVals[j]
            un = featFun(np.outer(feats[:, 0], y) + np.outer(feats[:, 1], np.ones((self.numCollPoints))))
            u += (self.featCoeffs[j] @ un) * self.indicatorVals[j]
        return u
    
    @staticmethod
    def exactSol(x: np.ndarray) -> np.ndarray:
        return np.sin(3 * np.pi * x + (3 * np.pi / 20)) * np.cos(2 * np.pi * x + (np.pi / 10)) + 2
    
    @line_profiler.profile
    def loss(self) -> np.ndarray:
        """
        The loss function calculating the square difference between the PDE value with the current approx solution and the r.h.s
        plus the same in the boundary conditions.

        This example case handels the (3.1) Helmoholts equation with lambda = -pi^2. (lambda = -25*pi^2.) -> then the exact solution is:
                u(x) = sin(3*pi*x + 3*pi/20)*cos(2*pi*x + pi/10) + 2

        Returns:
            float: value of the loss
        """
        c1 = 2.43177062311338
        c2 = 2.43177062311338

        a = self.lamb
        fx = -1/2 * (a + np.pi ** 2) * np.sin(np.pi * (self.collPoints[1:-1] + 1/20)) -1/2 * (a + 25 * np.pi ** 2) * np.sin(5 * np.pi * (self.collPoints[1:-1] + 1/20)) - 2 * a
        ux = self.evalCollocs()

        match self.partType:
            case "a":
                loss: np.ndarray = np.zeros((self.numCollPoints + 2 * (self.numParts - 1)))
                DDux = self.evalDDA(self.collPoints[1:-1])
            case _:
                loss: np.ndarray = np.zeros((self.numCollPoints))
                DDu = Derivative(self.evalSol, self.h, n = 2, method = "complex")
                DDux = DDu(self.collPoints[1:-1])

        loss[0] = self.lossWeightB[0 ] * (ux[0 ] - c1) # * np.pow(self.lossWeightB[0], 1/2) 
        loss[1] = self.lossWeightB[-1] * (ux[-1] - c2) # * np.pow(self.lossWeightB[1], 1/2)
        loss[2 : self.numCollPoints] = DDux - a * ux[1:-1] - fx

        if self.partType == "a":
            edges = np.linspace(0, 8, self.numParts + 1)[1:-1]
            for n in range(self.numParts - 1):
                loss[self.numCollPoints + n]                     = self.lossWeightE1[n] * (self.evalSolAPart(edges[n], n+1) - self.evalSolAPart(edges[n], n))
                loss[self.numCollPoints + self.numParts - 1 + n] = self.lossWeightE2[n] * (self.evalDAPart(edges[n], n+1) - self.evalDAPart(edges[n], n))
        return loss
    
    def optimFun(self, coeffs: np.ndarray) -> np.ndarray:
        """
        The loss as a function of the coefficient matrix

        Args:
            coeffs (np.ndarray): Coefficients for the feature functions in a numParts*numFuns vector

        Returns:
            float: The value of the loss
        """
        storeCoeffs = self.featCoeffs
        self.featCoeffs = np.reshape(coeffs, (self.featCoeffs.shape))
        loss = self.loss()
        self.featCoeffs = storeCoeffs
        return loss

    def plotFeats(self) -> None:
        sortedFreq = np.array([feats[feats[:, 0].argsort(), 0] for feats in self.featVals])
        sortedVals = np.array([np.abs(self.featCoeffs[i, feats[:, 0].argsort()]) for (feats, i) in zip(self.featVals, range(self.numParts))])

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        axs[0].imshow(sortedFreq, cmap='viridis', interpolation='nearest')
        axs[1].imshow(sortedVals, cmap='viridis', interpolation='nearest')
        plt.tight_layout()
        plt.show()

    @timeit
    def optimize(self, coeffs: np.ndarray|None = None) -> None:
        if coeffs == None:
            x0 = np.reshape(self.featCoeffs, (self.featCoeffs.size))
        elif coeffs.size == self.featCoeffs.size:
            x0 = np.reshape(coeffs, (self.featCoeffs.size))
        else:
            raise Exception("Starting coefficients are not in the correct format.")

        res = least_squares(self.optimFun, x0, jac = self.J, verbose=2, tr_solver = "exact", max_nfev = 200) # type: ignore
        self.featCoeffs = np.reshape(res.x, (self.featCoeffs.shape))
        return res

if __name__ == "__main__":
    main()