import numpy as np
from math import *
import cmath

class mechanicalDesign(object):
    def __init__(self):
        super(mechanicalDesign, self).__init__()

    def Motion(self,hipMotion,x,y,width,height):
        self.numPoints = len(hipMotion)

        hipMotionComplex = np.zeros(shape=(self.numPoints, 1), dtype=np.complex)
        for i in range(0, self.numPoints):
            hipMotionComplex[i] = complex(hipMotion[i, 0], -hipMotion[i, 1])

        uniformTiming = np.arange(0, 1+1/(self.numPoints-1), 1/(self.numPoints-1))
        timing = (2*(0.5*uniformTiming)**1.8-1)**3 + 1
        initialError = 1000
        fittingError = 10000

        bestSolError = np.arange((initialError+100 - 5), initialError, -10)  # 步长影响解的个数
        solution = np.zeros(shape=(len(bestSolError), 10), dtype=np.complex)

        maxCoupleRatio = 5
        firstLinkCoupleRatio = -1
        secondLinkCoupleRatio = np.arange(-maxCoupleRatio, maxCoupleRatio+1, 1)
        secondLinkCoupleRatio = np.delete(secondLinkCoupleRatio, 5)

        minDutyCycle = 80
        maxDutyCycle = 85
        dutyCycle = np.arange(minDutyCycle, maxDutyCycle+1, 1)

        For_Loop_Matrix = []
        for i in dutyCycle:
            for j in secondLinkCoupleRatio:
                For_Loop_Matrix.append([j, i])
        For_Loop_Matrix = np.array(For_Loop_Matrix)

        # fixPivotX = np.arange(170, 180, 2)
        # fixPivotY = np.arange(-450, -440, 2)  # 步长为1时运算量4倍
        fixPivotX = np.linspace(x, x + width, 5, endpoint=False)
        fixPivotY = np.linspace(-(y + height), -y, 5, endpoint=False)
        fixPivot = []
        for i in fixPivotY:
            for j in fixPivotX:
                fixPivot.append([j, i])
        fixPivot = np.array(fixPivot)

        FourierMatrix = np.zeros(shape=(self.numPoints, 2), dtype=np.complex)
        FourierMatrix_prestore = np.zeros(shape=(self.numPoints, 2), dtype=object)
        for i in range(0, len(For_Loop_Matrix)):
            for row in range(0, self.numPoints):
                FourierMatrix[row, 0] = cmath.exp(complex(0, (2*pi*For_Loop_Matrix[i, 1]/360*timing[row]*firstLinkCoupleRatio)))
                FourierMatrix[row, 1] = cmath.exp(complex(0, (2*pi*For_Loop_Matrix[i, 1]/360*timing[row]*For_Loop_Matrix[i, 0])))
            c = FourierMatrix.copy()
            FourierMatrix_prestore[i, 0] = c

        zeroMotionComplex = np.zeros(shape=(self.numPoints, 1), dtype=np.complex)
        higherMotionComplex = np.zeros(shape=(self.numPoints, 1), dtype=np.complex)

        for j in range(0, len(fixPivot)):
            for k in range(0, len(For_Loop_Matrix)):
                zeroMotionComplex[:, 0] = complex(fixPivot[j, 0], fixPivot[j, 1])
                higherMotionComplex = hipMotionComplex - zeroMotionComplex
                harmonics = np.dot(np.linalg.pinv(FourierMatrix_prestore[k, 0]), higherMotionComplex)
                U, fittingError, VT = np.linalg.svd(hipMotionComplex - np.dot(FourierMatrix_prestore[k, 0], harmonics) - zeroMotionComplex)
                maxError, errorIndex = max(bestSolError), np.argmax(bestSolError)  # argmax最大奇异值的位置
                # python中没有&&运算符  # abs-复数取绝对值
                if (fittingError < maxError) and min(abs(harmonics[1])) > 1 / 5 * abs(harmonics[0]):
                    bestSolError[errorIndex] = fittingError
                    solution[errorIndex, 0] = firstLinkCoupleRatio
                    solution[errorIndex, 1] = For_Loop_Matrix[k, 0]

                    solution[errorIndex, 3] = fixPivot[j, 0]
                    solution[errorIndex, 4] = fixPivot[j, 1]
                    solution[errorIndex, 5] = For_Loop_Matrix[k, 1]
                    solution[errorIndex, 6] = harmonics[0]
                    solution[errorIndex, 7] = harmonics[1]

                    solution[errorIndex, 9] = fittingError

        self.LinkLength = np.zeros(shape=(10, 2))
        LinkInitialPhase = np.zeros(shape=(10, 2))
        LinkInitialPhase_R1 = np.zeros(shape=(10, 1))
        LinkInitialPhase_R2 = np.zeros(shape=(10, 1))
        self.R0x_cordinate = np.zeros(shape=(10, 1))
        self.R0y_cordinate = np.zeros(shape=(10, 1))
        self.R1xMotion = np.zeros(shape=(10, self.numPoints))
        self.R1yMotion = np.zeros(shape=(10, self.numPoints))
        self.R2xMotion = np.zeros(shape=(10, self.numPoints))
        self.R2yMotion = np.zeros(shape=(10, self.numPoints))
        self.phase1 = np.zeros(shape=(10, self.numPoints))
        self.phase2 = np.zeros(shape=(10, self.numPoints))
        self.solutionMotion = np.zeros(shape=(10, 1),dtype=object)

        for n in range(0, len(solution)):
            FourierMatrix = np.zeros(shape=(self.numPoints, 2), dtype=np.complex)

            for row in range(0, self.numPoints):
                FourierMatrix[row, 0] = cmath.exp(complex(0, 2*pi*solution[n, 5]/360*timing[row]*solution[n, 0]))
                FourierMatrix[row, 1] = cmath.exp(complex(0, 2*pi*solution[n, 5]/360*timing[row]*solution[n, 1]))
            zeroMotionComplex[:, 0] = solution[n, 3]+complex(0, solution[n, 4].real)
            higherMotionComplex = hipMotionComplex-zeroMotionComplex
            harmonics = np.dot(np.linalg.pinv(FourierMatrix), higherMotionComplex)
            solutionMotion = np.dot(FourierMatrix, harmonics)+zeroMotionComplex
            self.solutionMotion[n,0] = solutionMotion
            self.LinkLength[n, 0] = abs(solution[n, 6])
            self.LinkLength[n, 1] = abs(solution[n, 7])
            LinkInitialPhase[n, 0] = np.degrees(cmath.phase(solution[n, 6]))  # cmath.phase返回弧度
            LinkInitialPhase[n, 1] = np.degrees(cmath.phase(solution[n, 7]))  # np.degrees返回角度
            self.R0x_cordinate[n, 0] = solution[n, 3].real
            self.R0y_cordinate[n, 0] = solution[n, 4].real

            LinkInitialPhase_R1[n, 0] = LinkInitialPhase[n, 0]
            LinkInitialPhase_R2[n, 0] = LinkInitialPhase[n, 1]

            for z in range(0, self.numPoints):
                self.R1xMotion[n, z] = self.R0x_cordinate[n, 0] + self.LinkLength[n, 0]*cos(radians(LinkInitialPhase_R1[n, 0]))
                self.R1yMotion[n, z] = self.R0y_cordinate[n, 0] + self.LinkLength[n, 0]*sin(radians(LinkInitialPhase_R1[n, 0]))  # cos(弧度)
                self.phase1[n, z] = LinkInitialPhase_R1[n, 0]  # 弧度转为度数
                LinkInitialPhase_R1[n, 0] = LinkInitialPhase_R1[n, 0] - abs(solution[n, 5])/self.numPoints
                # self.R2xMotion[n,z] = self.R1xMotion[n, z] + self.LinkLength[n,1]*cos(radians(LinkInitialPhase_R2[n, 0]))
                # self.R2yMotion[n, z] = self.R1yMotion[n, z] + self.LinkLength[n, 1] * sin(radians(LinkInitialPhase_R2[n, 0]))
                self.phase2[n, z] = LinkInitialPhase_R2[n, 0]
                LinkInitialPhase_R2[n, 0] = LinkInitialPhase_R2[n, 0] + solution[n, 1].real*abs(solution[n, 5])/self.numPoints
