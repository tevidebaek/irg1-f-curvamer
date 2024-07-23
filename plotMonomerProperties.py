import numpy as np
import matplotlib.pyplot as plt

#import the monomer data

data = np.loadtxt('monomer_properites.txt', delimiter=',', skiprows=1, dtype='str')

curvamer_radius = data.T[1].astype(float)
curvamer_length = data.T[2].astype(float)

plt.figure(figsize=(4,3))
plt.hist(curvamer_length, bins=20, ec='k', fc='w')
plt.xlim(0,200)

plt.xlabel('Radius of curvature [mono. len.]', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.tight_layout()

avg_len = np.median(curvamer_length)
length_cuts = np.where((curvamer_length>avg_len-10)&(curvamer_length<avg_len+10))

print(avg_len)

avg_len = 108




plt.figure(figsize=(4,3))
plt.hist(curvamer_radius[length_cuts]/avg_len, bins=20, ec='k', fc='w')

plt.xlim(0,1.2)
plt.xlabel('Radius of curvature [mono. len.]', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.tight_layout()

#to convert between radius of curvature and f (the fraction of the circle covered)
# s = r*\theta -> f = \theta/2*pi -> f = s/r/2/pi


circle_fraction = avg_len/curvamer_radius[length_cuts]/2/np.pi

print(np.average(circle_fraction), np.std(circle_fraction))

plt.figure(figsize=(4,3))
plt.hist(circle_fraction, bins=20, ec='k', fc='w')

plt.xlim(0,1)
plt.xlabel('f, circle fraction', fontsize=10)
plt.ylabel('Counts', fontsize=10)
plt.tight_layout()

plt.show()