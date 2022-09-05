
print("\n")
for str1 in open ( "galaxywitness/ansi.txt" ):
    print("\t\t\t" + str1, end = "")

for str2 in open ( "galaxywitness/ansiname.txt" ):
    print("\t\t\t" + str2, end = "")
    
def section():
    print("\n#########################################################################################\n")

print("\n\t\tTo Infinity... and Beyond!\n\n")
    
###########################################    

import time 
import os
import readline   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gudhi
from galaxywitness.witness_complex import WitnessComplex
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy import units as u

    

def plot_data_cloud():
    # plot point cloud
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter3D(witnesses[:, 0], witnesses[:, 1], witnesses[:, 2], s = 1, linewidths = 0.1)
    ax.scatter3D(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], s = 2, linewidths = 1)
    ax.set_xlabel('X, Mpc')
    ax.set_ylabel('Y, Mpc')
    ax.set_zlabel('Z, Mpc')

    if key_save == 'y':
        plt.savefig(path_to_save + '/plot_data_cloud.png', dpi = 200)
    plt.show()

def draw_diagrams_and_animation(key_anim):
    if key_anim == 'y':
        wc.animate_simplex_tree(path_to_save = path_to_save)
    
    wc.get_diagram(show = True, path_to_save = path_to_save) 
    wc.get_barcode(show = True, path_to_save = path_to_save)
    
def clustering(wc, path_to_save):
    t = time.time()
    tomato = wc.tomato()
    t = time.time() - t
    
    #tomato.plot_diagram()
    #tomato.n_clusters_ = int(input("Choose number of clusters: "))
    tomato.n_clusters_ = betti[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    ax.scatter3D(witnesses[:, 0], witnesses[:, 1], witnesses[:, 2], s = 1, c = tomato.labels_)
    ax.set_title("Tomato clustering")
    if path_to_save is not None:
        plt.savefig(path_to_save + f"/tomato.png", dpi = 200)
    plt.show()
    print(f"\U0001F345 done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
    


print("\nPreconfiguration:\n")
#readline.set_auto_history(True)
n_gal = int(input("Enter number of galaxies: "))
n_landmarks = int(input("Enter number of landmarks: "))
#n_jobs = int(input("Enter number of processes: "))

key_adv = input("Advanced configuration? [y/n]: ")
r_max = 20
first_witness = 0
tomato_key = 'y'
path = os.path.abspath('.') + '/data/result_glist_s.csv'
column_names = ['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']
isomap_eps = 0 
key_plot_cloud = 'y'
key_anim = 'y'
key_save = 'n'
if(key_adv) == 'y':
    key_plot_cloud = input("Do you want plot the point cloud? [y/n]: ")
    key_anim = input("Do you want watch the animation of witness filtration? [y/n]: ")
    key_save = input("Do you want save all plots to \033[01;32m./imgs\033[0m? [y/n]: ")
    r_max = int(input("Enter max value of filtration [-1 for None]: "))
    if r_max == -1:
        r_max = None
    print("\nChoose file with your data [.csv file]:")
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

section()

print(f"Loading data from \033[01;32m{path}\033[0m...")
t = time.time()
df = pd.read_csv(path)
t = time.time() - t

print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec. We have data about \033[01;32m{len(df)}\033[0m galaxies")

section()

if(key_adv) == 'y':
    print(f"Info about the handled table: \n\033[01;32m{df.info}\033[0m\n")
    
    list_names = list(df)
    
    print("\nChoose names of 3 columns for right ascension [RA], declination [Dec] and redshift [z]:")
    
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
    
section()
print("\nPreprocessing data and plot the point cloud...")

t = time.time()

witnesses = np.array(df[column_names].values[first_witness:n_gal + first_witness])
coord = SkyCoord(ra = witnesses[:, 0]*u.degree, dec = witnesses[:, 1]*u.degree, distance = Distance(z = witnesses[:, 2]))
witnesses = np.transpose(np.array(coord.cartesian.xyz), (1, 0))

landmarks = np.zeros((n_landmarks, 3))
landmarks_factor = int(n_gal/n_landmarks)
landmarks_idxs = np.zeros(n_landmarks, dtype = int)

for i, j in zip(range(0, n_gal, landmarks_factor), range(n_landmarks)): 
        landmarks[j, :] = witnesses[i, :]
        landmarks_idxs[j] = i
        
t = time.time() - t

print(f"Preprocessing done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.")

section()

if key_plot_cloud == 'y':
    print("\nTrying plot data cloud...")
    plot_data_cloud()
    print(f"Plot data cloud done \033[01;32m \u2714\033[0m")
    section()

print("Computing persistence with witness filtration...")

t = time.time()

witness_complex = gudhi.EuclideanStrongWitnessComplex(witnesses=witnesses, landmarks=landmarks)

simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=r_max**2, limit_dimension=3)
#wc.compute_simplicial_complex(d_max = 3, r_max = r_max,  n_jobs = n_jobs) ##simplex_tree = wc.simplex_tree
#print(simplex_tree.dimension()) 
wc = WitnessComplex(landmarks, witnesses, landmarks_idxs, simplex_tree = simplex_tree, isomap_eps = isomap_eps)
  
t = time.time() - t

betti = wc.get_persistence()
print(f"Persistence betti numbers: \n \033[01;32m{betti}\033[0m\n")
print(f"Computation done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
section()
print("Drawing persistence diagram and barcode...")
draw_diagrams_and_animation(key_anim)
print(f"Persistence diagram and barcode done \033[01;32m \u2714\033[0m")
section()

if tomato_key == 'y':
    print("ToMATo clustering...")
    clustering(wc, path_to_save)
    


            