#Name:-Mantu Kumar
#Roll no:-EE19B039



import numpy as np   #importing module sys
import matplotlib.pyplot as plt    #importing matplotlib to plot sophisticated Graphs
import scipy
import scipy.special as sp
# We are using "sys" module which contains that function.

scl=np.logspace(-1,-3,9)

def load(fittingfile):
    data = np.loadtxt(fittingfile)
    x = data[:,0]
    y = data[:,1:]
    return x,y

def Plot1(x,y):   #Defining a function Plot1
    plt.plot(x,y)
    scl=np.logspace(-1,-3,9)
    plt.title(r'A Plot of Differing Noise Levels')
    plt.xlabel(r'$t$',size=10)
    plt.ylabel(r'$f(t)+noise$',size=10)
    plt.legend(scl)
    plt.show()

def g(t,A=1.05,B=-0.105):#  Defining a function Taking Value of function
    return A*sp.jn(2,t)+B*t   #Giving Equation of Function

def Plotg(t):   #  Defining a Function g(t)
    plt.figure(0)
    plt.title('Original Plot')
    plt.plot(x,g(x))
    plt.xlabel(r'$t$',size=20)
    plt.ylabel(r'$f(t)$',size=20)
    plt.show()

def ErrorbarPlot(x,y,i):
    y_true = g(x)
    sigma = np.std(y[:,i]-y_true)
    plt.plot(x,y_true)
    plt.title(' Datapoints for sigma =' + str(scl[i]) + ' with error bars')
    plt.xlabel(r'$t$',size=20)
    plt.errorbar(x[::5],y[::5,i],sigma,fmt='ro')
    plt.show()

def generateP(x):
    M = np.zeros((x.shape[0],2))
    M[:,0] = sp.jn(2,x)
    M[:,1] = x
    return M

def error(x,AB):
    P = generateP(x)
    y_true = np.reshape(g(x),(101,1))
    y_pred = np.matmul(P,AB)
    return (np.square(y_pred - y_true)).mean()

def generateAB(i,j,step1 = 0.1,step2 = 0.01,Amin=0,Bmin = -0.2):
    AB = np.zeros((2,1))
    AB[0][0] = Amin +  step1 * i
    AB[1][0] = Bmin +step2 * j
    return AB

def Error_matrix(x,y,noise_index):
    try:
        y_noisy = np.reshape(y[:,noise_index],(101,1))
    except:
        y_noisy =np.reshape(g(x),(101,1))
    error = np.zeros((21,21))
    P = generateP(x)
    for i in range(21):
        for j in range(21):
            error[i,j] = np.square( np.matmul(P,generateAB(i,j)) - y_noisy).mean()
    return error

def Plot_contour(x,y):
    xp = np.linspace(0,2,21)
    yp = np.linspace(-0.2,0,21)
    X, Y = np.meshgrid(xp,yp)
    error = Error_matrix(x,y,0)
    CS = plt.contour(X,Y,error,[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5])
    plt.clabel(CS,CS.levels[:4], inline=1, fontsize=10)
    plt.title(' Error of Contour Plot ')
    plt.xlabel(r'$A$',size=20)
    plt.ylabel(r'$B$',size=20)
    plt.show()
    return

def estimateAB(M,b):
    return scipy.linalg.lstsq(M,b)

def error_pred(pred,true):
    return np.square(pred[0]-true[0]),np.square(pred[1]-true[1])


x,y = load("fitting.dat")
Plot1(x,y)   #ploting the defined function Plot1
Plotg(x)      #Ploting the defined Function g(x)
ErrorbarPlot(x,y,0)   #Ploting the Defined Function errorbarplot

AB = np.zeros((2,1))
AB[0][0] = 1.05
AB[1][0] = -0.105
print("Mean_square_error in calcutaion of M = ",error(x,AB))

Plot_contour(x,y)

prediction,error,_,_ = estimateAB(generateP(x),y[:,1])
print("Prediction  = ",prediction)

print("Error = ",error_pred(prediction,AB))

scl=np.logspace(-1,-3,9)
error_a = np.zeros(9)
error_b = np.zeros(9)
error_c = np.zeros(9)
for i in range(9):
    prediction,error,_,_ = estimateAB(generateP(x),y[:,i])
    error_a[i],error_b[i] = error_pred(prediction,AB)
    error_c[i] = error


plt.plot(scl,error_a,'r--')
plt.scatter(scl,error_a)
plt.plot(scl,error_b, 'b--')
plt.scatter(scl,error_b)
plt.legend(["A","B"])
plt.title("Error Variation with Noise")   #ploting the graph of Variation of Error With Noise
plt.xlabel(r'$\sigma_n$')
plt.ylabel(r'MS Error')
plt.show()

plt.loglog(scl,error_a,'r--')
plt.scatter(scl,error_a)
plt.loglog(scl,error_b, 'b--')
plt.scatter(scl,error_b)
plt.legend(["A","B"])
plt.title("Variation Of error with Noise on loglog scale") #ploting graph of error variation with noise on loglog scale
plt.xlabel(r'$\sigma_n$',size=20)
plt.ylabel(r'MS Error',size=20)
plt.show()

plt.loglog(scl,error_c, 'b--',basex = 20)
plt.scatter(scl,error_c)
plt.title("Error variation returned by Lstsq with Noise on loglog scale") #Error variation returned by least square with Noise on loglog scale
plt.xlabel(r'$\sigma_n$',size=20)
plt.ylabel(r'MS Error',size=20)
plt.show()


