import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#here we will plot some features of curvamers

data = pd.read_csv('CurvamerProperties.csv', delimiter=',')

t_mono = np.array(data['Mono Thick [px]'])
l_mono = np.array(data['Mono Length [px]'])
t_curv = np.array(data['Thickness [px]'])
p_in   = np.array(data['Inner Perim [px]'])
p_out  = np.array(data['Outer Perim [px]'])

#print(t_mono, l_mono, t_curv, p_in, p_out)

#things to calculate

stack_height = t_curv/t_mono

f_in = l_mono/p_in
f_out = l_mono/p_out

avg_mono_f = 0.31258173
std_mono_f = 0.07577269

avg_mono_inv_f = 1/avg_mono_f
std_mono_inv_f = std_mono_f/(avg_mono_f**2)

###############################################
# plot the stack height

plt.figure(figsize=(3,3))

bins = np.array(range(15))+0.5
plt.hist(stack_height, bins=bins, ec='k', fc='w')
plt.xlabel('stack height [mono.]', fontsize=10)
plt.ylabel('counts', fontsize=10)
plt.tight_layout()


###############################################
# plot the curvature of the inner and outer part of the stack

plt.figure(figsize=(3,3))
plt.scatter(stack_height, 1/f_in, c='r', alpha=0.2)
plt.scatter(stack_height, 1/f_out, c='b', alpha=0.2)

plt.axhline(avg_mono_inv_f, c='k')
plt.axhline(avg_mono_inv_f-std_mono_inv_f/2., c='k', ls='--')
plt.axhline(avg_mono_inv_f+std_mono_inv_f/2., c='k', ls='--')

plt.xlabel('stack height [mono.]', fontsize=10)
plt.ylabel('1/f', fontsize=10)

plt.xlim(0,15)
plt.ylim(0,7)
plt.tight_layout()

###############################################
# plot teh stack height versus the mean curvature

plt.figure(figsize=(3,3))
plt.scatter(stack_height, (1/2)*(1/f_in+1/f_out), color='k', alpha=0.2)

plt.xlabel('stack height [mono.]', fontsize=10)
plt.ylabel(r'$R_{avg}$ [mono. len.]', fontsize=10)

plt.xlim(0,15)
plt.ylim(0,5)


plt.show()