
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import rc 
rc('text', usetex=True) #requires installing Latex (install or comment out)
						#sudo apt-get install texlive-full

#TO RUN THIS SCRIPT: 
#      $python plotter.py dataAndPlots/fileName_dev.txt dataAndPlots/fileName_train.txt

#-----------------------------------------------------------------------------------------------
#Plots made so far:
# python3 plotter.py dataAndPlots/20180604-081431_dev_resnet18_224.txt dataAndPlots/20180604-081431_train_resnet18_224.txt
# python3 plotter.py dataAndPlots/20180605-010950_dev_resnet18_decay_224.txt dataAndPlots/20180605-010950_train_resnet18_decay_224.txt
# (7 epochs) python3 plotter.py dataAndPlots/20180605-061011_dev_resnet18_decay_adam_224.txt dataAndPlots/20180605-061011_train_resnet18_decay_adam_224.txt
# python3 plotter.py dataAndPlots/20180605-075524_dev_resnet50_224.txt dataAndPlots/20180605-075524_train_resnet50_224.txt
# python3 plotter.py dataAndPlots/20180606-010128_dev_resnet18_decay_adam_224.txt dataAndPlots/20180606-010128_train_resnet18_decay_adam_224.txt
# python3 plotter.py dataAndPlots/20180606-120447_dev_resnet18_decay_16.txt dataAndPlots/20180606-120447_train_resnet18_decay_16.txt
# python3 plotter.py dataAndPlots/20180606-235007_dev_resnet18_augment_224.txt dataAndPlots/20180606-235007_train_resnet18_augment_224.txt
#-----------------------------------------------------------------------------------------------

#This file will generate the following plots and 
#store them in the folder "dataAndPlots" with 
#the same name as the datafile:
#      - loss vs. epoch
#      - prec@1 vs. epoch
#
#For both training and dev

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12} #24 for poster

plt.rc('font', **font)

#Data to plot
data_dev = np.genfromtxt("./" + sys.argv[-2])
data_train = np.genfromtxt("./" + sys.argv[-1]) #sys.argv[-1] looks at last terminal input 
epochs = data_train[:,0] 
loss_train = data_train[:,3]
loss_dev = data_dev[:,2]
acc_train = data_train[:,4]
acc_dev = data_dev[:,3] 

#Loss vs. Epoch
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(epochs,loss_train,'m-',label='train')
plt.plot(epochs,loss_dev,'g-',label='dev')
plt.legend(loc='best')
# plt.axis('equal')
# plt.xticks(range(1, 1 + int(max(epochs))))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.title('Loss vs. Epoch')

#Precision vs. Epoch
plt.subplot(1,2,2)
plt.plot(epochs, acc_train,'m-',label='train')
plt.plot(epochs,acc_dev,'g-',label='dev')
plt.legend(loc='best')
# plt.axis('equal')
# plt.xticks(range(1, 1 + int(max(epochs))))
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.tight_layout()
plt.title('Precision vs. Epoch')

#Show and save
plt.savefig( sys.argv[-1] + ".png")
plt.show()
