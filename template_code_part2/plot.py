import matplotlib.pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10]
MAP1=[0.604,0.655,0.666,0.660,0.654,0.644,0.636,0.629,0.621,0.606]
precision1=[0.604,0.513,0.458,0.420,0.376,0.352,0.322,0.301,0.284,0.270]
recall1=[0.102,0.167,0.211,0.255,0.282,0.312,0.331,0.348,0.365,0.386]
fscore1=[0.168,0.237,0.268,0.294,0.298,0.306,0.302,0.299,0.296,0.295]
nDCG1=[0.482,0.451,0.434,0.423,0.412,0.411,0.406,0.404,0.403,0.401]
MAP2 = [0.657,0.704,0.715,0.712,0.697,0.689,0.681,0.674,0.661,0.656]
precision2=[0.657,0.560,0.497,0.443,0.407,0.374,0.353,0.332,0.311,0.292]
recall2=[0.112,0.185,0.240,0.276,0.311,0.337,0.363,0.385,0.402,0.419]
fscore2=[0.184,0.261,0.301,0.314,0.325,0.327,0.331,0.329,0.324,0.318]
nDCG2=[0.525,0.496,0.473,0.462,0.454,0.450,0.448,0.447,0.446,0.444]
precision3=[0.698,0.591,0.525,0.465,0.427,0.405,0.371,0.348,0.325,0.310]
recall3=[0.118,0.192,0.251,0.290,0.324,0.361,0.381,0.404,0.421,0.441]
fscore3=[0.192,0.272,0.317,0.331,0.341,0.352,0.347,0.345,0.340,0.337]
MAP3=[0.697,0.731,0.754,0.751,0.741,0.721,0.715,0.703,0.694,0.684]
nDCG3=[0.567,0.520,0.498,0.480,0.472,0.473,0.467,0.464,0.463,0.461]
# figure, axis = plt.subplots(3, 2)


plt.subplot(1, 3, 1)
plt.plot(x,precision3,marker='o',label="LSA",color='b')
plt.plot(x,precision2,marker='o',label="VSM2",color='g')
plt.legend()
plt.title("precision vs k")


plt.subplot(1, 3, 2)
plt.plot(x,recall3,marker='o',label="LSA",color='b')
plt.plot(x,recall2,marker='o',label="VSM2",color='g')
plt.legend()
plt.title("recall vs k")

plt.subplot(1, 3, 3)
plt.plot(x,fscore3,marker='o',label="LSA",color='b')
plt.plot(x,fscore2,marker='o',label="VSM2",color='g')
plt.legend()
plt.title("fscore vs k")

plt.savefig("VSM2 vs LSA (precison,recall,fscore)")

plt.subplot(1, 2, 1)
plt.plot(x,MAP3,marker='o',label="LSA",color='b')
plt.plot(x,MAP2,marker='o',label="VSM2",color='g')
plt.legend()
plt.title("MAP vs k")

plt.subplot(1, 2, 2)
plt.plot(x,nDCG3,marker='o',label="LSA",color='b')
plt.plot(x,nDCG2,marker='o',label="VSM2",color='g')
plt.legend()
plt.title("nDCG vs k")
plt.savefig("VSM2 vs LSA (MAP,nDCG)")

# plt.show()