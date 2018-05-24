
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

data_filename = sys.argv[-1]
data = np.genfromtxt(data_filename)

plt.figure(1)
plot(data[4,:], data[2,:])
plt.title('Loss vs. time')
plt.xlabel('time')
plt.ylabel('loss')
plt.savefig("./plots/." + data_filename + "_loss"+".png")

plt.figure(2)
plot(data[5,:], data[2,:])
plt.title('Accuracy vs. time')
plt.xlabel('time')
plt.ylabel('prec@1')
plt.savefig("./plots/." + data_filename + "_accuracy"+".png")

plt.show()