import Surface_confined_inference as sci
import numpy as np
import os
import matplotlib.pyplot as plt
freqs=[2.98,3.99, 7.99]
str_freqs=[str(x) for x in freqs]
dec_amount=64
loc="/home/henryll/Documents/Experimental_data/Jamie/set1/"
files=os.listdir(loc)
for i in range(0, len(freqs)):
    file=[x for x in files if str_freqs[i] in x and "voltage" in x]
    currentfile=[x for x in files if str_freqs[i] in x and "current" in x][0]
    print(file)
    data=np.loadtxt(os.path.join(loc,file[0]))
    current=np.loadtxt(os.path.join(loc,currentfile))[::dec_amount, 1]
    
    time=data[::dec_amount,0]
    potential=data[::dec_amount,1]
    plt.plot(time, potential, label=str_freqs[i])
    print(sci.infer.get_input_parameters(time, potential, current, "FTACV", runs=20, plot_results=True))
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Potential (V)")
plt.show()
#