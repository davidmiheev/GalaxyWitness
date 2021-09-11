
print("\n")
for str in open ( "data/ansi.txt" ):
    print("\t\t\t" + str, end = "")

for str1 in open ( "data/ansiname.txt" ):
    print("\t\t" + str1, end = "")

import time 
import os
import readline   
import torch
import matplotlib.pyplot as plt
import pandas as pd
from galaxywitness.witness_complex import WitnessComplex

print("\n     Let's go, first preconfiguration\n")

n_gal = int(input("Enter number of galaxies: "))
n_landmarks = int(input("Enter number of landmarks: "))
n_jobs = int(input("Enter number of processes: "))

key = input("Do you want compute only zeroth \u2119\u210d? [y/n]: ")
key_anim = input("Do you want watch the animation of witness filtration (works only if |landmarks| <= 50)? [y/n]: ")


print("\n#########################################################################################\n")

path = os.path.abspath('.') + '/data/result_glist_s.csv'
print(f"Load data from \033[01;32m{path}\033[0m...")
t = time.time()
df = pd.read_csv(path)
t = time.time() - t

print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec. We have data about \033[01;32m{df['z_gal'].size}\033[0m galaxies")

print("\n#########################################################################################\n")
witnesses = torch.tensor(df[['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']].values[:n_gal])

landmarks = torch.zeros(n_landmarks, 3)
landmarks_factor = int(n_gal/n_landmarks)

for i, j in zip(range(0, n_gal, landmarks_factor), range(0, n_landmarks)): 
        landmarks[j, :] = witnesses[i, :]

# plot point cloud
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
n_points_for_plot_1 = min(n_gal, 10000)
n_points_for_plot_2 = min(n_landmarks, 1000)

ax.scatter(witnesses[:n_points_for_plot_1, 0], witnesses[:n_points_for_plot_1, 1], witnesses[:n_points_for_plot_1, 2], linewidths=0.1)
ax.scatter(landmarks[:n_points_for_plot_2, 0], landmarks[:n_points_for_plot_2, 1], landmarks[:n_points_for_plot_2, 2], linewidths=3.5)

plt.show()

print("\n#########################################################################################\n")
print("Compute persistence with witness complex and draw persitence diagram and barcode...")

wc = WitnessComplex(landmarks, witnesses)

t = time.time()
if key == 'n':
    wc.compute_simplicial_complex(d_max = 2, create_simplex_tree = True, create_metric = False, n_jobs = n_jobs)#simplex_tree = wc.simplex_tree print(simplex_tree.dimension()) 
    
if key == 'y':
    t = time.time()
    wc.compute_metric_optimized(n_jobs = n_jobs)
    wc.compute_1d_simplex_tree()

t = time.time() - t

if key_anim == 'y' and n_landmarks <= 50:
    wc.animate_simplex_tree()
wc.get_diagram(show = True, path_to_save = None) 
wc.get_barcode(show = True, path_to_save = None)
print(f"Computation done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
#print(wc.landmarks_dist, end="\n#####\n")


            