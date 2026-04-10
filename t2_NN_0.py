# 
# --- modeFRONTIER Response Surface ----------------
# Code Created by
# modeFRONTIER  - (c) ESTECO S.p.A.
# modeFRONTIER Version modeFRONTIER 2025R1.1 16.7.4 b20250217
# Date Fri Apr 10 12:51:24 JST 2026
# Project Name mf結果①.prj
# Operating System Windows 11 10.0 amd64
# Java (SDK/JRE) Version 21.0.5
# Java Vendor Eclipse Adoptium
# Java Vendor URL https://adoptium.net/
# User Name nakay
# 
# 
# --- DISCLAIMER - Please do not erase -------------
# NO WARRANTY ON RSM CODE
# The Response Surface Methodology ("RSM") is a code which, due to the nature of machine learning based predictive models,
# may provide inaccurate output or otherwise not always produce the intended results. Therefore it should not be relied on
# as the sole basis to implement a design, whose incorrect implementation could result in injury to person or property.
# This code is not intended for use in any inherently dangerous applications, including applications which may create a risk
# of personal injury. If you use this code without reserve, you take full responsibility to grant all appropriate fail-safe,
# backup, redundancy, and other measures to ensure its safe use.
# 
# ESTECO makes to the Customer no warranty, express or implied, with reference to the compliance of the RSM code with a particular use.
# 
# Furthermore, ESTECO:
# (i) makes no warranty, express or implied, on the merchantability and fitness of the RSM code for a particular purpose,
# (ii) does not warrant that the operation or other use of the RSM code be uninterrupted or error free or will cause damage or
# disruption to the Customer’s data, computers or networks.
# 
# 
# --------------------------------------------------
# x[0] corresponds to variable R1
# x[1] corresponds to variable R2
# x[2] corresponds to variable R3
# x[3] corresponds to variable R4
# x[4] corresponds to variable R5
# x[5] corresponds to variable RT
# --------------------------------------------------
# 
# 
# --------------------------------------------------
# Response Surface Name : t2_NN_0
# Response Surface Type : Neural Networks
# --------------------------------------------------
# 

import math
import csv
import sys

class t2_NN_0:
    def __init__(self):
        self.n_input = 3
        # load data from file
        try:
            with open('t2_NN_0.csv') as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader)
                next(filereader)
                self.x_range = [[0 for _ in range(2)] for _ in range(3)]
                for i in range(3):
                    self.x_range[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.y_range = [0 for _ in range(2)]
                for i in range(2):
                    self.y_range[i] = float(next(filereader)[0])
                next(filereader)
                self.out_range = [0 for _ in range(2)]
                for i in range(2):
                    self.out_range[i] = float(next(filereader)[0])
                next(filereader)
                self.w1 = [[0 for _ in range(3)] for _ in range(10)]
                for i in range(10):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(10)]
                for i in range(10):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(10)] for _ in range(1)]
                for i in range(1):
                    self.w2[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b2 = [0 for _ in range(1)]
                for i in range(1):
                    self.b2[i] = float(next(filereader)[0])
                next(filereader)
                csvfile.close()
        except OSError:
            print("ERROR: cannot open the data file")
            sys.exit(1)
        except StopIteration:
            pass

    def evaluate(self, x):
        # check input
        if len(x) != 6:
            print("ERROR - Wrong Input Vector Length")
            return math.nan
        # keep only important input variables
        xx = [x[1], x[3], x[5]]
        # warning: variable x[0] is ignored
        # warning: variable x[2] is ignored
        # warning: variable x[4] is ignored

        # normalize input
        xn = [0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]) / (self.x_range[i][1] - self.x_range[i][0])

        # perform computations
        n1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]
        y1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            exp = math.exp(-2.0 * n1[i])
            if exp == math.inf:
                y1[i] = -1.0
            else:
                y1[i] = (1.0 - exp)/(1.0 + exp)
        n2 = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            n2[i] = self.b2[i]
            for j in range(len(self.w2[0])):
                n2[i] += self.w2[i][j] * y1[j]
        yn = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            yn[i] = n2[i]
        # scale output
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0])/(self.out_range[1] - self.out_range[0]) * (yn[0] - self.out_range[0])
        return y


    def get_input_variable_names(self):
        return ["R1", "R2", "R3", "R4", "R5", "RT"]

    def get_output_variable_name(self):
        return "t2"

