
import numpy as np
import matplotlib.pyplot as plt
import sys
# from matplotlib import rc 
# rc('text', usetex=True) #requires installing Latex (install or comment out)
						  #sudo apt-get install texlive-full

#TO RUN THIS SCRIPT: 
#      $python plotter.py dataAndPlots/fileName_dev.txt dataAndPlots/fileName_train.txt

#This file will generate the following plots and 
#store them in the folder "dataAndPlots" with 
#the same name as the datafile:
#      - loss vs. epoch
#      - prec@1 vs. epoch
#
#For both training and dev

#Data to plot
data_dev = np.genfromtxt("./" + sys.argv[-2])
data_train = np.genfromtxt("./" + sys.argv[-1]) #sys.argv[-1] looks at last terminal input 
epochs = data_train[:,0] 
loss_train = data_train[:,3]
loss_dev = data_dev[:,2]
acc_train = data_train[:,5]
acc_dev = data_dev[:,4] 

#Loss vs. Epoch
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(epochs,loss_train,label='train')
plt.plot(epochs,loss_dev,label='dev')
plt.legend(loc='best')
plt.xticks(range(1, 1 + int(max(epochs))))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')

#Accuracy vs. Epoch
plt.subplot(1,2,2)
plt.plot(epochs, acc_train,label='train')
plt.plot(epochs,acc_dev,label='dev')
plt.legend(loc='best')
plt.xticks(range(1, 1 + int(max(epochs))))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')

#Show and save
plt.savefig( sys.argv[-1] + ".png")
plt.show()
