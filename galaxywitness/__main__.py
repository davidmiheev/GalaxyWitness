
print("\n")
for str1 in open ( "galaxywitness/ansi.txt" ):
    print("\t\t\t" + str1, end = "")

for str2 in open ( "galaxywitness/ansiname.txt" ):
    print("\t\t" + str2, end = "")
    
###########################################    

import time 
import os
import readline   
import torch
import numpy as np
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

key = input("Do you want compute only zeroth \033[01;32m\u2119\u210d\033[0m? [y/n]: ")
key_anim = input("Do you want watch the animation of witness filtration? [y/n]: ")
key_save = input("Do you want save all plots to \033[01;32m./imgs\033[0m? [y/n]: ")
key_adv = input("Advanced configuration? [y/n]: ")
r_max = 50
first_witness = 0
tomato_key = 'n'
path = os.path.abspath('.') + '/data/result_glist_s.csv'
column_names = ['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']
isomap_eps = 0 
if(key_adv) == 'y':
    r_max = int(input("Enter max value of filtration [-1 for None]: "))
    if r_max == -1:
        r_max = None
    
    data_tables = os.walk('./data')
    print("\n\t---------- data -----------")
    for _, _, elem in data_tables:
        for name in elem:
            print(f"\t{elem.index(name)+1} <- {name}")
    print("\t---------------------------\n")
    table_num = int(input(f"Enter number of your table [1-{len(elem)}]: "))
    path = os.path.abspath('.') + '/data/' +  elem[table_num - 1]
    isomap_eps = float(input("Enter\033[01;32m isomap\033[0m parameter [0 - don't compute isomap metric]: "))
    tomato_key = input("Do you want run\033[01;32m tomato\033[0m clustering? [y/n]: ") 
    #cosmology = input("Enter cosmology model: ")

path_to_save = None
time_list = list(time.localtime())
time_str = ''
for i in range(6):
    time_str += str(time_list[i]) 

if key_save == 'y':
    path_to_save = os.path.abspath('.') + '/imgs/' + time_str + f"-{n_gal}-{n_landmarks}"
    if(not os.path.isdir('imgs')):
        os.mkdir('imgs')
    os.mkdir(path_to_save)

        
print("\n#########################################################################################\n")

print(f"Load data from \033[01;32m{path}\033[0m...")
t = time.time()
df = pd.read_csv(path)
t = time.time() - t

print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec. We have data about \033[01;32m{len(df)}\033[0m galaxies")

print("\n#########################################################################################\n")
if(key_adv) == 'y':
    print(f"Info about the handled table: \n\033[01;32m{df.info}\033[0m\n")
    
    list_names = list(df)
    print("\n\t---------- column names -----------")
    for elem in list_names:
        if list_names.index(elem) != 0:
            print(f"\t{list_names.index(elem)} <- {elem}")
    print("\t-----------------------------------\n")
    column_nums = []
    for i in range(3):
        column_nums.append(int(input(f"Choose number of column #{i+1} of 3, from list above (column names): "))) 
    column_names = [list(df)[column_nums[0]], list(df)[column_nums[1]], list(df)[column_nums[2]]]
    first_witness = int(input(f"Enter index of first witness [0-{df[column_names[2]].size-n_gal}]: "))
print("\nPreprocessing data and plot the point cloud...")
t = time.time()
witnesses = torch.tensor(df[column_names].values[first_witness:n_gal + first_witness])
coord = SkyCoord(ra = witnesses[:, 0]*u.degree, dec = witnesses[:, 1]*u.degree, distance = Distance(z = witnesses[:, 2]))
witnesses = (torch.tensor(coord.cartesian.xyz)).transpose(0,1)

landmarks = torch.zeros(n_landmarks, 3)
landmarks_factor = int(n_gal/n_landmarks)
landmarks_idxs = np.zeros(n_landmarks, dtype = int)

for i, j in zip(range(0, n_gal, landmarks_factor), range(n_landmarks)): 
        landmarks[j, :] = witnesses[i, :]
        landmarks_idxs[j] = i
        
t = time.time() - t
print(f"Preprocessing done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.")
# plot point cloud
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter3D(witnesses[:MAX_N_PLOT, 0], witnesses[:MAX_N_PLOT, 1], witnesses[:MAX_N_PLOT, 2], s = 3, linewidths = 0.1)
ax.scatter3D(landmarks[:MAX_N_PLOT, 0], landmarks[:MAX_N_PLOT, 1], landmarks[:MAX_N_PLOT, 2], s = 6, linewidths = 3)
ax.set_xlabel('X, Mpc')
ax.set_ylabel('Y, Mpc')
ax.set_zlabel('Z, Mpc')

if key_save == 'y':
    plt.savefig(path_to_save + '/plot_data_cloud.png', dpi = 200)
plt.show()


print("\n#########################################################################################\n")
print("Compute persistence with witness complex and draw persitence diagram, barcode...")

t = time.time()

wc = WitnessComplex(landmarks, witnesses, landmarks_idxs, isomap_eps = isomap_eps)

if key == 'n':
    wc.compute_simplicial_complex(d_max = 2, r_max = r_max,  n_jobs = n_jobs)#simplex_tree = wc.simplex_tree print(simplex_tree.dimension())  
    
if key == 'y':
    t = time.time()
    wc.compute_metric_optimized(n_jobs = n_jobs)
    wc.compute_1d_simplex_tree(r_max = r_max)

t = time.time() - t
if key_anim == 'y':
    wc.animate_simplex_tree(path_to_save = path_to_save)
    
wc.get_diagram(show = True, path_to_save = path_to_save) 
wc.get_barcode(show = True, path_to_save = path_to_save)
betti = wc.get_persistence()
print(f"Persistence betti numbers: \n \033[01;32m{betti}\033[0m\n")
print(f"Computation done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")

if tomato_key == 'y':
    t = time.time()
    tomato = wc.tomato()
    t = time.time() - t
    
    #tomato.plot_diagram()
    #tomato.n_clusters_ = int(input("Choose number of clusters: "))
    tomato.n_clusters_ = betti[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.scatter3D(witnesses[:, 0], witnesses[:, 1], witnesses[:, 2], s = 3, c = tomato.labels_)
    ax.set_title("Tomato clustering")
    if path_to_save is not None:
        plt.savefig(path_to_save + f"/tomato.png", dpi = 200)
    plt.show()
    print(f"\U0001F345 done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
#print(wc.landmarks_dist, end="\n#####\n")


            