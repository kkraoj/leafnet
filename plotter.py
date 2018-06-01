
import numpy as np
import matplotlib.pyplot as plt
import sys
# from matplotlib import rc
# rc('text', usetex=True)

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
its_train = data_train[:,1] #iterations
its_dev = data_dev[:,1] #iterations
batch_size_train = np.max(its_train) + 1
batch_size_dev = np.max(its_dev) + 1
epochs_train = data_train[:,0] + its_train/batch_size_train
epochs_dev = data_dev[:,0] + its_dev/batch_size_dev
loss_train = data_train[:,4]
loss_dev = data_dev[:,3]
acc_train = data_train[:,5]
acc_dev = data_dev[:,4] 

#Loss vs. Epoch
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(epochs_train, loss_train,label='train')
plt.plot(epochs_dev,loss_dev,label='dev')
plt.legend(loc='best')
plt.xticks(range(1,1+int(max(data_train[:,0]))))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. Epoch')

#Accuracy vs. Epoch
plt.subplot(1,2,2)
plt.title('Accuracy vs. Epoch')
plt.plot(epochs_train, loss_train,label='train')
plt.plot(data_dev[:,0]/50,loss_dev,label='dev')
plt.legend(loc='best')
plt.xticks(range(1,1+int(max(data_train[:,0]))))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. Epoch')

#Show and save
plt.savefig( sys.argv[-1] + ".png")
plt.show()
