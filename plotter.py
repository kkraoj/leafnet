
import numpy as np
import matplotlib.pyplot as plt
import sys

#To run this script: 
#      $python plotter.py datafile.txt

#This file will generate the following plots and 
#store them in the folder "performance_plots" with 
#the same name as the datafile:
#      - loss vs. time
#      - prec@1 vs. time

data_filename = "./epoch_data/" + sys.argv[-1]
data = np.genfromtxt(data_filename)
time = np.cumsum(data[2,:])

plt.figure(1)
plt.plot(time, data[4,:])
plt.title('Loss vs. time')
plt.xlabel('time')
plt.ylabel('loss')
plt.savefig("./plots/" + sys.argv[-1] + "_loss"+".png")

plt.figure(2)
plt.plot(time, data[5,:])
plt.title('Accuracy vs. time')
plt.xlabel('time')
plt.ylabel('prec@1')
plt.savefig("./plots/" + sys.argv[-1] + "_accuracy"+".png")

plt.show()