import numpy as np
from math import *
import cmath
import random
import scipy.stats as stats

class nBar(object):
    def __init__(self):
        self.num_machanical = 0
        self.length = []
        self.phase = []
        self.role_num = []
        self.R0x = []
        self.R0y = []
        self.error = []
        self.fixedPivot = []
        self.velocity = []
        self.endVelocity = []
        self.endAcceleration = []
        super(nBar, self).__init__()

    def clear(self):
        self.num_machanical = 0
        self.R0x.clear()
        self.R0y.clear()
        self.length.clear()
        self.phase.clear()
        self.role_num.clear()
        self.error.clear()
        self.fixedPivot.clear()
        self.velocity.clear()
        self.endVelocity.clear()
        self.endAcceleration.clear()

    def randomTiming(self,num):
        lower, upper = 0, 1  # 最大值与最小值
        mu, sigma = 0, 1  # 标准正态分布
        np.random.seed(10)
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
        timing = X.rvs(num)
        timing = np.sort(timing)
        return timing

    def Motion(self, hipMotion, barNum, x, y, width, height,maxlength,minlength,timenum):
        self.maxlength = int(maxlength)
        self.minlength = int(minlength)
        self.barNum = barNum
        self.numPoints = len(hipMotion)
        self.hipMotion = hipMotion
        self.solution = np.zeros(shape=(3, 2 * self.barNum + 4), dtype=np.complex)
        self.hipMotionComplex = np.zeros(shape=(self.numPoints, 1), dtype=np.complex)
        for i in range(0, self.numPoints):
            self.hipMotionComplex[i] = complex(hipMotion[i, 0], -hipMotion[i, 1])

        uniformTiming = np.arange(0, 1+1/(self.numPoints-1), 1/(self.numPoints-1))

        self.errorFactor = np.zeros(shape=(2 * self.barNum, 1))

        if timenum == 1:
            self.timing = (2 * (0.5 * uniformTiming) ** 1.5 - 1) ** 3 + 1
        elif timenum == 2:
            self.timing = (2 * (0.5 * uniformTiming) ** 1.5 - 1) ** 3 + 1
        else:
            self.timing = (2 * (0.5 * uniformTiming) ** 1.5 - 1) ** 3 + 1

        self.initialError = 1000

        self.firstLinkCoupleRatio = -1

        minDutyCycle = 80
        maxDutyCycle = 80
        dutyCycle = np.arange(minDutyCycle, maxDutyCycle+1, 2)
        ratiolist = self.cal(barNum,dutyCycle)
        var = np.meshgrid(*ratiolist)
        var_ravel = [v.ravel() for v in var]
        For_Loop_Matrix = [v[:] for v in var_ravel]
        For_Loop_Matrix = np.array(For_Loop_Matrix)

        minDutyCycle = 70
        maxDutyCycle = 70
        dutyCycle = np.arange(minDutyCycle, maxDutyCycle + 1, 2)
        ratiolist = self.cal(barNum, dutyCycle)
        var = np.meshgrid(*ratiolist)
        var_ravel = [v.ravel() for v in var]
        For_Loop_Matrix_2 = [v[:] for v in var_ravel]
        For_Loop_Matrix_2 = np.array(For_Loop_Matrix_2)

        minDutyCycle = 80
        maxDutyCycle = 130
        dutyCycle = np.arange(minDutyCycle, maxDutyCycle, 10)
        ratiolist = self.cal(barNum, dutyCycle)
        var = np.meshgrid(*ratiolist)
        var_ravel = [v.ravel() for v in var]
        For_Loop_Matrix_3 = [v[:] for v in var_ravel]
        For_Loop_Matrix_3 = np.array(For_Loop_Matrix_3)

        fixPivotX = np.linspace(x, x + width, 5)
        fixPivotY = np.linspace(-(y + height), -y, 5)

        [a, b] = np.meshgrid(fixPivotX, fixPivotY)
        a = a.ravel()  # 展开成一维
        b = b.ravel()
        fixPivot = [a[:], b[:]]
        self.fixPivot = np.array(fixPivot)

        self.fourier(For_Loop_Matrix, 0)
        self.fourier(For_Loop_Matrix_2, 1)
        self.fourier(For_Loop_Matrix_3, 2)
        self.mechPara()

        for i in range(3):
            self.role_num.append(barNum)
            self.length.append(self.LinkLength[i].copy())
            self.phase.append(self.phase_[i].copy())
            self.R0x.append(self.R0x_cordinate[i][0])
            self.R0y.append(self.R0y_cordinate[i][0])
            self.error.append(self.solution[i][-1].real)
            self.endVelocity.append(self.vxy[i,:].copy())
            self.endAcceleration.append(self.axy[i,:].copy())
        self.num_machanical += 3

    def cal(self, linkNum, dutyCycle):
        ratiolist = []
        for n in range(2,linkNum+1):
            LinkCoupleRatio = np.arange(-5, 5, 1)
            LinkCoupleRatio = np.delete(LinkCoupleRatio, 5)
            ratiolist.append(LinkCoupleRatio)
        ratiolist.append(dutyCycle)
        return ratiolist

    def fourier(self,For_Loop_Matrix,errorIndex):
        FourierMatrix = np.zeros(shape=(self.numPoints, self.barNum), dtype=np.complex)
        FourierMatrix_prestore = np.zeros(shape=(For_Loop_Matrix.shape[1], self.barNum), dtype=object)
        svd_prestore = np.zeros(shape=(For_Loop_Matrix.shape[1], 3), dtype=object)
        for i in range(0, For_Loop_Matrix.shape[1]):
            for row in range(0, self.numPoints):
                FourierMatrix[row, 0] = cmath.exp(complex(0, (2*pi*For_Loop_Matrix[self.barNum-1,i]/360*self.timing[row]*self.firstLinkCoupleRatio)))
                for k in range(1,self.barNum):
                    FourierMatrix[row, k] = cmath.exp(complex(0, (2*pi*For_Loop_Matrix[self.barNum-1,i]/360*self.timing[row]*For_Loop_Matrix[k-1,i])))
            c = FourierMatrix.copy()
            FourierMatrix_prestore[i, 0] = c

            U, S, V = np.linalg.svd(FourierMatrix_prestore[i, 0])
            svd_prestore[i, 0] = U
            svd_prestore[i, 1] = S
            svd_prestore[i, 2] = np.transpose(V)

        zeroMotionComplex = np.zeros(shape=(self.numPoints, 1), dtype=np.complex)
        bestSolError = 1000

        for j in range(0, self.fixPivot.shape[1]):
            for k in range(0, For_Loop_Matrix.shape[1]):
                zeroMotionComplex[:, 0] = complex(self.fixPivot[0, j], self.fixPivot[1, j])
                higherMotionComplex = self.hipMotionComplex - zeroMotionComplex
                harmonics = np.dot(np.linalg.pinv(FourierMatrix_prestore[k, 0]), higherMotionComplex)
                harmonics_svd = np.dot(svd_prestore[k, 2], self.errorFactor)
                harmonics_sum = harmonics + harmonics_svd

                U, fittingError, VT = np.linalg.svd(
                    self.hipMotionComplex - np.dot(FourierMatrix_prestore[k, 0], harmonics) - zeroMotionComplex)
                maxError = bestSolError
                if (fittingError < maxError) and min(abs(harmonics)) > self.minlength and max(abs(harmonics)) < self.maxlength:
                    bestSolError = fittingError
                    self.solution[errorIndex,0] = self.firstLinkCoupleRatio
                    for i in range(0,self.barNum):
                        self.solution[errorIndex,i+1] = For_Loop_Matrix[i, k]
                    self.solution[errorIndex, self.barNum + 1] = self.fixPivot[0, j]
                    self.solution[errorIndex, self.barNum + 2] = self.fixPivot[1, j]
                    for i in range(0,self.barNum):
                        self.solution[errorIndex, self.barNum+3+i] = harmonics[i]
                    self.solution[errorIndex, 2*self.barNum+3] = fittingError


        velocity = []
        for l in range(self.barNum):
            velocity.append(self.solution[errorIndex, l].real)
        self.velocity.append(velocity)

    def mechPara(self):
        self.LinkLength = np.zeros(shape=(3, self.barNum))
        LinkInitialPhase = np.zeros(shape=(3, self.barNum))
        self.R0x_cordinate = np.zeros(shape=(3, 1))
        self.R0y_cordinate = np.zeros(shape=(3, 1))

        self.phase_ = np.zeros(shape=(3, self.barNum), dtype=object)
        phase1 = np.zeros(shape=(1, self.numPoints))
        phase2 = np.zeros(shape=(1, self.numPoints))
        self.InitialPhase = []

        vx = np.zeros(shape=(1,self.numPoints))
        vy = np.zeros(shape=(1, self.numPoints))
        vi = np.zeros(shape=(1, self.numPoints))
        vj = np.zeros(shape=(1, self.numPoints))
        ax = np.zeros(shape=(1,self.numPoints))
        ay = np.zeros(shape=(1, self.numPoints))
        ai = np.zeros(shape=(1, self.numPoints))
        aj = np.zeros(shape=(1, self.numPoints))
        self.vxy = np.zeros(shape=(3, self.numPoints))
        self.axy = np.zeros(shape=(3, self.numPoints))

        for n in range(0, 3):
            self.R0x_cordinate[n, 0] = self.solution[n, self.barNum + 1].real
            self.R0y_cordinate[n, 0] = self.solution[n, self.barNum + 2].real
            for i in range(0,self.barNum):
                LinkInitialPhase[n, i] = np.degrees(cmath.phase(self.solution[n, self.barNum+3+i]))
                self.LinkLength[n, i] = abs(self.solution[n, self.barNum+3+i])

            self.InitialPhase.append(LinkInitialPhase[n,:])
            for z in range(0, self.numPoints):
                phase1[0, z] = LinkInitialPhase[n, 0]
                LinkInitialPhase[n, 0] = LinkInitialPhase[n, 0] - abs(self.solution[n, self.barNum]) / self.numPoints
            phase = phase1.copy()
            self.phase_[n, 0] = phase[0]

            for i in range(1,self.barNum):
                for z in range(0,self.numPoints):
                    phase2[0, z] = LinkInitialPhase[n, i]
                    LinkInitialPhase[n, i] = LinkInitialPhase[n,i] + self.solution[n, i].real * abs(self.solution[n, self.barNum]) / self.numPoints
                phase = phase2.copy()
                self.phase_[n, i] = phase[0]

            for k in range(0, self.numPoints):
                for j in range(0,self.barNum):
                    phasei = self.phase_[n, j]

                    vi[0, k] = -self.solution[n, j].real*self.LinkLength[n,j]*sin(radians(phasei[k]))
                    vj[0, k] = self.solution[n, j].real * self.LinkLength[n, j] * cos(radians(phasei[k]))
                    vx[0, k] = vx[0, k] + vi[0, k]
                    vy[0, k] = vy[0, k] + vj[0, k]
                    ai[0, k] = -self.solution[n, j].real**2 * self.LinkLength[n, j] * cos(radians(phasei[k]))
                    aj[0, k] = -self.solution[n, j].real**2 * self.LinkLength[n, j] * sin(radians(phasei[k]))
                    ax[0, k] = ax[0, k] + ai[0, k]
                    ay[0, k] = ay[0, k] + aj[0, k]

                self.vxy[n, k] = sqrt(vx[0, k] ** 2 + vy[0, k] ** 2)
                self.axy[n, k] = sqrt(ax[0, k] ** 2 + ay[0, k] ** 2)
