import matplotlib.pyplot as plt, numpy as np, copy,pandas as pd, math, os,seaborn as sns, pygmt,random,copy
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def read(filename,num_clo):
    '''
        File read function.
        :param filename: File path
        :param num_clo: File cloumn numbers
        :return: 2D array
        '''
    with open(filename) as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        lines = [float(ii) for i in lines for ii in i]
        lines = np.array(lines).reshape(-1,num_clo)
    return lines



#Each column of the ss.txt should be: Lon, Lat, Depth, Wavelength band depth, Wavelength band GA,  Wavelength band VGG, GA, VGG
file = r"ss.txt"
data = read(file, 8)

df = pd.DataFrame(data)

fun = LinearRegression()
fun.fit(df.values[:,3].reshape(-1,1),df.values[:,4].reshape(-1,1))

fun1 = LinearRegression()
fun1.fit(df.values[:,3].reshape(-1,1),df.values[:,5].reshape(-1,1))


data1 = read(r"ss1.txt", 8)
temp1 = fun.coef_ * data1[:,2]
temp2 = fun1.coef_ * data1[:,2]

temp = np.concatenate([data1[:,[0,1,2,6,7]], data1[:,6].reshape(-1,1) - temp1.reshape(-1,1), data1[:,7].reshape(-1,1) - temp2.reshape(-1,1)], 1)

np.savetxt(r"Training dataset.txt", temp, fmt='%f', delimiter='   ', newline='\n')

