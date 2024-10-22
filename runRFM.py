import numpy as np
import matplotlib.pyplot as plt
import randomFeatureMethod4 as rfm
from numdifftools import Derivative
import datetime

np.random.seed(17)
options1 =  (2,  50, 102,  30, "sin", "B", 1e-2)
options2 =  (4,  50, 202,  30, "sin", "B", 1e-3)
options3 =  (8,  50, 402,  30, "sin", "B", 1e-4)
options4 =  (16, 50, 802,  30, "sin", "B", 1e-4)
options5 =  (32, 50, 1602, 30, "sin", "B", 1e-4)
options6 =  (2,  50, 102,  30, "sin", "A", 1e-4)
options7 =  (4,  50, 202,  30, "sin", "A", 1e-5)
options8 =  (8,  50, 402,  30, "sin", "A", 1e-5)
options9 =  (16, 50, 802,  30, "sin", "A", 1e-4)
options10 = (32, 50, 1602, 30, "sin", "A", 1e-4)
options11 = (2, 100, 202,  50, "sin", "A", 1e-4)
options12 = (2, 400, 802,  15, "tanh","B", 1e-4)
numParts, numFuns, numCollPoints, randRange, featFunType, partType, h = options12

rfm1 = rfm.rfm1d(numParts, numFuns, numCollPoints, randRange, featFunType, partType, h, rescaling = True)
print("Initialization done.")
res = rfm1.optimize()

# Generate values for plotting
X = np.linspace(0, 8, 16000)

# Calculating the values of the numerical solution 
Yrfm = rfm1.evalSol(X)
YD = rfm1.evalDA(X)
YDD = rfm1.evalDDA(X)
YaD = Derivative(rfm1.exactSol,  1e-6, method="complex", n = 1)
YaDD = Derivative(rfm1.exactSol, 1e-6, method="complex", n = 2)

# Calculating the values of the exact solution
Yexact = rfm1.exactSol(X)
Err = Yexact - Yrfm
m = np.max(np.abs(Err))
print(f"{"L∞ error:":<9} {m:.2e}")

ErrD = YD - YaD(X) # type: ignore
mD = np.max(np.abs(ErrD))
ErrD /= mD
print(f"{"mD:":<9} {mD:.2e}")

ErrDD = YDD - YaDD(X) # type: ignore
mDD = np.max(np.abs(ErrDD))
ErrDD /= mDD
print(f"{"mDD:":<9} {mDD:.2e}")

# Plotting the result
# Plot the function
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 12))
if partType == "A":
    numCollPoints += 2 * (numParts - 1)
fig.suptitle(f"M: {numParts * numFuns}, N: {numCollPoints}, featFun: {featFunType}, type: {partType}, h: {h:.1e}, L∞ error: {m:.2e}", fontsize=16)
axs[2].plot(X, YDD, label = "rfmDD")
axs[2].plot(X, YaDD(X), label = "exactDD", linestyle='--')
axs[2].plot(X, ErrDD * 0.8 * np.max(np.abs(YDD)), label = "DD error", linestyle=':') # type: ignore
axs[2].grid(True)
axs[2].set_title("Second Derivative values")
axs[2].legend()

axs[1].plot(X, YD, label = "rfmD")
axs[1].plot(X, YaD(X), label = "exactD", linestyle='--')
axs[1].plot(X, ErrD * 0.8 * np.max(np.abs(YD)), label = "D error", linestyle=':') # type: ignore
axs[1].grid(True)
axs[1].set_title("Derivative values")
axs[1].legend()

axs[0].plot(X, Yrfm, label = "rfm")
axs[0].plot(X, Yexact, label = "exact", linestyle='--')
axs[0].plot(X, Err * 0.8 / m + 2, label = "error", linestyle=':')
axs[0].grid(True)
axs[0].set_title("Function values")
axs[0].legend()
filename = f"plot_{datetime.datetime.now().strftime('%d_%H-%M-%S')}.png"
plt.savefig("../plots/" + filename)
plt.show()

rfm1.plotFeats()

