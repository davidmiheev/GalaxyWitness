
print("\n")
for str1 in open ( "data/ansi.txt" ):
    print("\t\t\t" + str1, end = "")

for str2 in open ( "data/ansiname.txt" ):
    print("\t\t" + str2, end = "")
    
###########################################    

import time 
import os
import readline   
import torch
import matplotlib.pyplot as plt
import pandas as pd
from galaxywitness.witness_complex import WitnessComplex
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy import units as u

MAX_N_PLOT = 20000

print("\n     Let's go, first preconfiguration\n")

n_gal = int(input("Enter number of galaxies: "))
n_landmarks = int(input("Enter number of landmarks: "))
n_jobs = int(input("Enter number of processes: "))

key = input("Do you want compute only zeroth \u2119\u210d? [y/n]: ")
key_anim = input("Do you want watch the animation of witness filtration? [y/n]: ")
key_save = input("Do you want save all plots to \033[01;32m./imgs\033[0m? [y/n]: ")

time_list = list(time.localtime())
time_str = ''
for i in range(0, 6):
    time_str += str(time_list[i]) 
path_to_save = None

if key_save == 'y':
    path_to_save = os.path.abspath('.') + '/imgs/' + time_str
    if(not os.path.isdir('imgs')):
        os.mkdir('imgs')
    os.mkdir(path_to_save)

        
print("\n#########################################################################################\n")

path = os.path.abspath('.') + '/data/result_glist_s.csv'
print(f"Load data from \033[01;32m{path}\033[0m...")
t = time.time()
df = pd.read_csv(path)
t = time.time() - t

print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec. We have data about \033[01;32m{df['z_gal'].size}\033[0m galaxies")

print("\n#########################################################################################\n")
print("Preprocessing data and plot the data cloud...")
witnesses = torch.tensor(df[['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']].values[:n_gal])
c = SkyCoord(ra = witnesses[:, 0]*u.degree, dec = witnesses[:,1]*u.degree, distance = Distance(z = witnesses[:, 2]))
witnesses = (torch.tensor(c.cartesian.xyz)).transpose(0,1)

landmarks = torch.zeros(n_landmarks, 3)
landmarks_factor = int(n_gal/n_landmarks)

for i, j in zip(range(0, n_gal, landmarks_factor), range(0, n_landmarks)): 
        landmarks[j, :] = witnesses[i, :]

# plot point cloud
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(witnesses[:MAX_N_PLOT, 0], witnesses[:MAX_N_PLOT, 1], witnesses[:MAX_N_PLOT, 2], linewidths=0.1)
ax.scatter(landmarks[:MAX_N_PLOT, 0], landmarks[:MAX_N_PLOT, 1], landmarks[:MAX_N_PLOT, 2], linewidths=3.5)
ax.set_xlabel('X, Mpc')
ax.set_ylabel('Y, Mpc')
ax.set_zlabel('Z, Mpc')

if key_save == 'y':
    plt.savefig(path_to_save + '/plot_data_cloud.png', dpi = 200)
plt.show()


print("\n#########################################################################################\n")
print("Compute persistence with witness complex and draw persitence diagram, barcode...")

wc = WitnessComplex(landmarks, witnesses)

t = time.time()
if key == 'n':
    wc.compute_simplicial_complex(d_max = 2, create_metric = False, r_max = 90, create_simplex_tree = True,  n_jobs = n_jobs)#simplex_tree = wc.simplex_tree print(simplex_tree.dimension())  
    
if key == 'y':
    t = time.time()
    wc.compute_metric_optimized(n_jobs = n_jobs)
    wc.compute_1d_simplex_tree(r_max = 90)

t = time.time() - t

if key_anim == 'y':
    wc.animate_simplex_tree(path_to_save = path_to_save)
    
wc.get_diagram(show = True, path_to_save = path_to_save) 
wc.get_barcode(show = True, path_to_save = path_to_save)
print(f"Computation done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
#print(wc.landmarks_dist, end="\n#####\n")


            