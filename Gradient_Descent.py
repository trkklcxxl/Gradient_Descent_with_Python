import numpy as np
import matplotlib.pylab as plt 



def PlotData(x_train,y_train,y_predicted,x_label,y_label,title):
    # Plot our model prediction
    plt.plot(x_train, y_predicted, c='b',label='Our Prediction')
    plt.scatter(x_train,y_train,marker='x',c='r')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
#PlotData(x_train,y_train,y_predicted,"house size","rent price","housing size")

def Gradient_Descent(x_train,y_train):
    m=len(x_train)
    w=0
    b=0
    learning_rate=0.08
    iterations=1000
    for i in range(iterations):
        y_predicted=w*x_train+b
        cost=sum([val**2 for  val in(y_predicted-y_train)])/m
        temp_w=w-learning_rate*(sum((y_predicted-y_train)*x_train))/m
        temp_b=b-learning_rate*sum((y_predicted-y_train))/m
        w=temp_w
        b=temp_b
        print ("m {}, b {}, cost {} iteration {}".format(m,b,cost, i))
    PlotData(x_train,y_train,y_predicted,"house size","rent price","housing size")    

# in this example x axis is size of houses and y axis is rent prices of houses
#x_train=np.array([75,90,96,100,120,130,170,260]) #size of houses 
#y_train=np.array([3500,4100,5000,4500,6000,4500,6000,10000]) # rent price of houses

x_train = np.array([1,2,3,4,5])
y_train = np.array([5,7,9,11,13])

Gradient_Descent(x_train,y_train)
