print("\n")
for str in open ( "data/ansi.txt" ):
    print("\t\t\t" + str, end = "")

for str1 in open ( "data/ansiname.txt" ):
    print("\t\t" + str1, end = "")

import time 
import os   
import torch
import matplotlib.pyplot as plt
import pandas as pd
from galaxywitness.witness_complex import WitnessComplex

print("\n     Let's go, first preconfiguration\n")
n_gal = int(input("Enter number of galaxies: "))
n_proc = int(input("Enter number of processes: "))
n_landmarks = int(input("Enter number of landmarks: "))
print("############################\n")

path = os.path.abspath('.') + '/data/result_glist_s.csv'
print(f"Load data from \033[01;32m{path}\033[0m...")
t = time.time()
df = pd.read_csv(path)
t = time.time() - t

print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec. We have data about " , end = "")
print(f"{df['z_gal'].size} galaxies")
print("\n#############################################################\n")
witnesses = torch.tensor(df[['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']].values[:n_gal])
#indexes = torch.rand(int(n_gal/3), 1)
landmarks = torch.zeros(n_landmarks, 3)
landmarks_factor = int(n_gal/n_landmarks)
j = 0
for i in range(0, n_gal, landmarks_factor):
    if j < n_landmarks: 
        landmarks[j, :] = witnesses[i, :]
        j += 1

#print(witnesses)

# plot point cloud
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
n_points_for_plot_1 = min(n_gal, 10000)
n_points_for_plot_2 = min(n_landmarks, 1000)

ax.scatter(witnesses[:n_points_for_plot_1, 0], witnesses[:n_points_for_plot_1, 1], witnesses[:n_points_for_plot_1, 2])
ax.scatter(landmarks[:n_points_for_plot_2, 0], landmarks[:n_points_for_plot_2, 1], landmarks[:n_points_for_plot_2, 2])

plt.show()

print("\n#############################################################\n")
print("Compute persistence with witness complex and draw persitence diagram and barcode...")
t = time.time()
# computation of a witness simplicial complex and construct barcode
wc = WitnessComplex(landmarks, witnesses)

wc.compute_simplicial_complex(d_max = 1, create_simplex_tree = True, create_metric = True, n_jobs = n_proc)

simplex_tree = wc.simplex_tree #how to retrieve the simplex_tree
t = time.time() - t
wc.get_diagram(show = True, path_to_save = None) # allows to directly plot a persistence diagram
wc.get_barcode(show = True, path_to_save = None)
print(f"Computation done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")


#max_ra = float(landmarks[:, 0].max())
#min_ra = float(landmarks[:, 0].min())
#max_de = float(landmarks[:, 1].max())
#min_de = float(landmarks[:, 1].min())
#max_z = float(landmarks[:, 2].max())
#min_z = float(landmarks[:, 2].min())
#scale_ra = (max_ra - min_ra)/2
#scale_de = (max_de - min_de)/2
#scale_z = (max_z - min_z)/2
#med_ra = (max_ra + min_ra)/2
#med_de = (max_de + min_de)/2
#med_z = (max_z + min_z)/2
