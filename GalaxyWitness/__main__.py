import os
import time
import requests
import sys

################# Banner ########################
print("\n")
try:
    for str1 in open ( "GalaxyWitness/ansi.txt" ):
        print("\t\t\t" + str1, end = "")

    for str2 in open ( "GalaxyWitness/ansiname.txt" ):
        print("\t\t\t" + str2, end = "")
except Exception:
    print("\033[01;32mGalaxyWitness\033[0m\n")
    print("\033[01;33mWarning: Can't load the banner!\033[0m")

print("\n\t\tTo Infinity... and Beyond!\n\n")
print("Loading...")
#################################################

import webbrowser
import readline
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import gudhi

import urllib.request
from tqdm import tqdm
import ssl

from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy import units as u

from GalaxyWitness.witness_complex import WitnessComplex
from GalaxyWitness.downloader import DownloadProgressBar

MAX_DIM = 3

def pause(t = 1):
    time.sleep(t) 

def section():
    print("\n\n")

def plot_data_cloud(witnesses, landmarks, key_save, path_to_save):
    # plot point cloud
    fig = go.Figure(data = [go.Scatter3d(x=witnesses[:, 0],
                                        y=witnesses[:, 1], 
                                        z=witnesses[:, 2], 
                                        mode='markers', 
                                        marker=dict(size=1, color='blue')), 
                            go.Scatter3d(x=landmarks[:, 0], 
                                        y=landmarks[:, 1], 
                                        z=landmarks[:, 2], 
                                        mode='markers', 
                                        marker=dict(size=2, color='orange'))])
                                        
    fig.update_layout(scene = dict(xaxis_title='X, Mpc', 
                                   yaxis_title='Y, Mpc', 
                                   zaxis_title='Z, Mpc'))
    
    if key_save == 'y':
        fig.write_image(path_to_save + '/plot_data_cloud.pdf')
        
    fig.show()

def draw_diagrams_and_animation(fil_complex, key_anim, path_to_save, key_fig):
    if key_anim == 'y':
        if key_fig == 'plotly':
            fil_complex.animate_simplex_tree_plotly(path_to_save = path_to_save)
        else:
            fil_complex.animate_simplex_tree(path_to_save = path_to_save)
    
    fil_complex.get_diagram(show = True, path_to_save = path_to_save) 
    fil_complex.get_barcode(show = True, path_to_save = path_to_save)
    
def clustering(fil_complex, path_to_save):
    print("ToMATo clustering...")
    t = time.time()
    tomato = fil_complex.tomato()
    t = time.time() - t
    
    #tomato.plot_diagram()
    fig = go.Figure(data = [go.Scatter3d(x=fil_complex.witnesses[:, 0], 
                                         y=fil_complex.witnesses[:, 1], 
                                         z=fil_complex.witnesses[:, 2], 
                                         mode='markers', 
                                         marker = dict(size=1, color=tomato.labels_))])
                                         
    fig.update_layout(scene = dict(xaxis_title = "X, Mpc", 
                                   yaxis_title = "Y, Mpc", 
                                   zaxis_title = "Z, Mpc"))

    if path_to_save is not None:
        fig.write_image(path_to_save + "/tomato.pdf")
        
    fig.show()
    
    print(f"\033[F\U0001F345 clustering... done\033[01;32m \u2714\033[0m\
    in \033[01;32m{t}\033[0m sec.\n")

def download_prepared_datasets():
    # временное решение, пока не подготовили датасеты
    ssl._create_default_https_context = ssl._create_unverified_context
    print("Choose one of available datasets:")
    print("1. result_glist_s")
    print("2. another dataset")
    input_dataset = int(input())
    if input_dataset == 1:
        url = "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_glist_s.csv"
        fname = "result_glist_s.csv"

        os.chdir("./data")
        with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=fname, reporthook=t.update_to)
        os.chdir("..")
    else:
        print("will be implemented soon...")

def preconfiguration():
    print(f"\nSystem information: \033[01;32m{os.uname()}\033[0m")
    print("\nPreconfiguration:\n")
    prepared = input(" > Do you want to use prepared datasets? [y/n]: ")

    if prepared == "y":
        download_prepared_datasets()
    
    print("\nChoose file with your data [.csv file]:")
    
    data_tables = os.walk('./data')
    
    print(data_tables)  
    print("\n\t---------- data -----------")
    for _, _, elem in data_tables:
        for name in elem:
            print(f"\t{elem.index(name)+1} <- {name}")
    print("\t---------------------------\n")
        
    table_num = int(input(f" > Enter number of your table [1-{len(elem)}]: "))
    path = os.path.abspath('.') + '/data/' + elem[table_num - 1]

    print(f"Loading data from \033[01;32m{path}\033[0m...")
    t = time.time()
    df = pd.read_csv(path)
    t = time.time() - t

    print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\
    We have data about \033[01;32m{len(df)}\033[0m galaxies.\n")
    
    return df

def main():
 
    try:
        df = preconfiguration()
    except ValueError: 
        raise Exception("\033[01;31mFolder 'data' does not exist or empty!\033[0m")
       
    doc_key = input(" > Do you want to open documentation? [y/n]: ")
    if doc_key == 'y':
        url = 'file://' + os.path.abspath('.') + '/docs/build/html/index.html'
        webbrowser.open(url, new=2)
        
    #readline.set_auto_history(True)
    n_gal = int(input(f" > Enter number of galaxies [10-{len(df)}]: "))
    
    type_of_complex = input(" > Enter type of complex [witness/alpha]: ")
    
    if type_of_complex == "witness":
        n_landmarks = int(input(f" > Enter number of landmarks [10-{n_gal}]: "))
    else:
        n_landmarks = n_gal
            
    #n_jobs = int(input("Enter number of processes: "))

    key_adv = input(" > Advanced configuration? [y/n]: ")
    r_max = 7.5
    first_witness = 0
    tomato_key = 'y'
    column_names = ['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']
    isomap_eps = 0 
    key_plot_cloud = 'y'
    key_anim = 'y'
    key_save = 'n'
    key_complex_type = 'gudhi'
    key_fig = 'mpl'
    if(key_adv) == 'y':
        key_plot_cloud = input(" > Do you want plot the point cloud? [y/n]: ")
        key_anim = input(" > Do you want watch the animation of witness filtration? [y/n]: ")
        if key_anim == 'y':
            key_fig = input(" > What will we use for the animation of a filtered complex? [plotly(more slow, but more cool)/mpl(matplotlib)]: ")
        key_save = input(" > Do you want save all plots to \033[01;32m./imgs\033[0m? [y/n]: ")
    
        key_complex_type = input(" > What type of simplicial complex will we use? [gudhi/custom]: ")
        r_max = float(input(" > Enter max value of filtration[\033[01;32m usually \u2264 15\033[0m, the more the slower calculate]: "))
        tomato_key = input(" > Do you want run\033[01;32m tomato\033[0m clustering? [y/n]: ")
        if r_max == -1:
            r_max = None
        
        isomap_eps = float(input(" > Enter\033[01;32m isomap\033[0m parameter [0 - don't compute isomap metric]: "))
     
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

    if(key_adv) == 'y':
        section()
        print(f"Info about the handled table: \n\033[01;32m{df.info}\033[0m\n")
    
        list_names = list(df)
        pause()
        print("\nChoosing names of 3 columns for right ascension [RA], declination [Dec] and redshift [z]:")
    
        print("\n\t---------- column names -----------")
        for elem in list_names:
            if list_names.index(elem) != 0:
                print(f"\t{list_names.index(elem)} <- {elem}")
                
        print("\t-----------------------------------\n")
    
        column_nums = []
        
        for i in range(3):
            column_nums.append(int(input(f" > Choose number of column #{i+1} of 3, from list above (column names): ")))
             
        column_names = [list(df)[column_nums[0]], list(df)[column_nums[1]], list(df)[column_nums[2]]]
        first_witness = int(input(f" > Enter index of first witness [0-{df[column_names[2]].size-n_gal}]: "))
    
    section()
        
    print("Preprocessing data and plot the point cloud...")

    t = time.time()

    witnesses = np.array(df[column_names].values[first_witness:n_gal + first_witness])
    
    coord = SkyCoord(
                    ra = witnesses[:, 0]*u.degree, 
                    dec = witnesses[:, 1]*u.degree, 
                    distance = Distance(z = witnesses[:, 2])
                    )
                    
    witnesses = np.transpose(np.array(coord.cartesian.xyz), (1, 0))

    landmarks = np.zeros((n_landmarks, 3))
    landmarks_factor = int(n_gal/n_landmarks)
    landmarks_idxs = np.zeros(n_landmarks, dtype = int)

    for i, j in zip(range(0, n_gal, landmarks_factor), range(n_landmarks)): 
        landmarks[j, :] = witnesses[i, :]
        landmarks_idxs[j] = i
        
    t = time.time() - t

    print(f"\033[FPreprocessing data and plot the point cloud... done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.")
    
    pause()
    section()

    if key_plot_cloud == 'y':
        print("Trying plot data cloud...")
        plot_data_cloud(witnesses, landmarks, key_save, path_to_save)
        print(f"\033[FTrying plot data cloud... done \033[01;32m\u2714\033[0m")
        pause()
        section()

    print("Computing persistence with witness filtration...")

    t = time.time()
    witness_complex = WitnessComplex(landmarks, witnesses, landmarks_idxs, isomap_eps = isomap_eps)

    if key_complex_type == 'custom':
        witness_complex.compute_simplicial_complex(d_max = MAX_DIM, r_max = r_max, custom=True)
    else:
        if type_of_complex == "witness":
            witness_complex.compute_simplicial_complex(d_max = MAX_DIM, r_max = r_max)
            # witness_complex = gudhi.EuclideanStrongWitnessComplex(witnesses=witnesses, landmarks=landmarks)
            # simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=r_max**2, limit_dimension = MAX_DIM)
        else:
            acomplex = gudhi.AlphaComplex(points=witnesses)
            simplex_tree = acomplex.create_simplex_tree(max_alpha_square=r_max**2)
            witness_complex.external_simplex_tree(simplex_tree) # TODO: make separate class for alpha (filtered) complex
            
    if key_complex_type == 'gudhi':
        magnitude_level = (r_max**2)/2.0
    else:
        magnitude_level = r_max/2.0
        
    simplex_tree = witness_complex.simplex_tree

    betti = witness_complex.get_persistence_betti(dim = MAX_DIM, magnitude = magnitude_level)

    t = time.time() - t
    
    print(f"\033[FComputing persistence with witness filtration... done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
    pause()
    
    print(f"The \033[01;32msimplex tree\033[0m constructed \033[01;32m\u2714\033[0m") 
    print(f"\t\033[01;32msimplex tree stats:\n\t dim: {simplex_tree.dimension()}\033[0m")
    print(f"\t\033[01;32m number of vertices (landmarks): {simplex_tree.num_vertices()}\033[0m")
    print(f"\t\033[01;32m total number of simplices: {simplex_tree.num_simplices()}\033[0m")
    print(f"\t\033[01;32m persistence betti numbers: {betti}\033[0m")
    
    section()
    
    print("\nDrawing persistence diagram and barcode...")

    draw_diagrams_and_animation(witness_complex, key_anim, path_to_save, key_fig)

    print(f"\033[F\033[F\033[FDrawing persistence diagram and barcode... done \033[01;32m \u2714\033[0m")
    pause()
    
    section()

    if tomato_key == 'y':
        clustering(witness_complex, path_to_save)
        
    
if __name__ == '__main__':
    print("\033[FLoading... done \033[01;32m\u2714\033[0m")
    pause()
    main()



            