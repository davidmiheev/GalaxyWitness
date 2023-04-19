import os
import time
import ssl
import webbrowser
import site


################# Banner ########################
print("\n")
try:
    with open(site.getsitepackages()[0] + "/galaxywitness/ansi.txt", encoding="utf-8") as ansi:
        for str1 in ansi:
            print("\t\t\t" + str1, end="")

    with open(site.getsitepackages()[0] + "/galaxywitness/ansiname.txt", encoding="utf-8") as ansiname:
        for str2 in ansiname:
            print("\t\t\t" + str2, end="")
except OSError:
    print("\033[01;32mGalaxyWitness\033[0m\n")
    print("\033[01;33mWarning: Can't load the banner! Dev version\033[0m")

print("\n\t\tTo Infinity... and Beyond!\n\n")
print("Loading...")
#################################################

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

import numpy as np
import plotly.graph_objects as go
import pandas as pd


from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy import units as u


from galaxywitness.base_complex import BaseComplex
from galaxywitness.witness_complex import WitnessComplex
from galaxywitness.alpha_complex import AlphaComplex
from galaxywitness.rips_complex import RipsComplex
from galaxywitness.datasets import Dataset
from galaxywitness.clusterization import Clusterization

session = PromptSession(history=InMemoryHistory(),
                        auto_suggest=AutoSuggestFromHistory())

MAX_DIM = 3


def pause(t=1):
    time.sleep(t)


def section():
    print("\n\n")


def plot_data_cloud(witnesses, landmarks, key_save, path_to_save):
    # plot point cloud
    fig = go.Figure(data=[go.Scatter3d(x=witnesses[:, 0],
                                       y=witnesses[:, 1],
                                       z=witnesses[:, 2],
                                       mode='markers',
                                       marker=dict(size=1, color='blue')),
                          go.Scatter3d(x=landmarks[:, 0],
                                       y=landmarks[:, 1],
                                       z=landmarks[:, 2],
                                       mode='markers',
                                       marker=dict(size=2, color='orange'))])

    fig.update_layout(scene=dict(xaxis_title='X, Mpc',
                                 yaxis_title='Y, Mpc',
                                 zaxis_title='Z, Mpc'))

    if key_save == 'y':
        fig.write_image(path_to_save + '/plot_data_cloud.pdf')

    fig.show()


def plot_data_cloud_alpha(landmarks, key_save, path_to_save):
    # plot point cloud
    fig = go.Figure(data=[go.Scatter3d(x=landmarks[:, 0],
                                       y=landmarks[:, 1],
                                       z=landmarks[:, 2],
                                       mode='markers',
                                       marker=dict(size=1, color='orange'))])

    fig.update_layout(scene=dict(xaxis_title='X, Mpc',
                                 yaxis_title='Y, Mpc',
                                 zaxis_title='Z, Mpc'))

    if key_save == 'y':
        fig.write_image(path_to_save + '/plot_data_cloud.pdf')

    fig.show()


def draw_diagrams_and_animation(fil_complex, key_anim, path_to_save, key_fig):
    if key_anim == 'y':
        if key_fig == 'plotly':
            fil_complex.animate_simplex_tree_plotly(path_to_save=path_to_save)
        else:
            fil_complex.animate_simplex_tree(path_to_save=path_to_save)

    fil_complex.get_diagram(show=True, path_to_save=path_to_save)
    # fil_complex.get_barcode(show=True, path_to_save=path_to_save)


def clustering(fil_complex, points, r_max, path_to_save):
    print("ToMATo clustering...")
    t = time.time()
    tomato = fil_complex.tomato(r_max)
    t = time.time() - t

    # tomato.plot_diagram()
    fig = go.Figure(data=[go.Scatter3d(x=points[:, 0],
                                       y=points[:, 1],
                                       z=points[:, 2],
                                       mode='markers',
                                       marker=dict(size=1, color=tomato.labels_))])

    fig.update_layout(scene=dict(xaxis_title="X, Mpc",
                                 yaxis_title="Y, Mpc",
                                 zaxis_title="Z, Mpc"))

    if path_to_save is not None:
        fig.write_image(path_to_save + "/tomato.pdf")

    fig.show()

    print(f"\033[F\U0001F345 clustering... done\033[01;32m \u2714\033[0m\
    in \033[01;32m{t}\033[0m sec.\n")
    return tomato


def download_prepared_datasets():
    ssl._create_default_https_context = ssl._create_unverified_context
    print("Choose one of available datasets:")
    print("\n\t---------- datasets -----------")
    print("\t\033[01;32m[1] <-> Galaxies_400K\033[0m")
    print("\t\033[01;32m[2] <-> Galaxies_1KK\033[0m")
    print("\t\033[01;32m[3] <-> go to ./data folder \033[0m")
    print("\t--------------------------------\n")
    input_dataset = int(session.prompt(' > Enter number of dataset: ', auto_suggest=AutoSuggestFromHistory()))
    name = ''
    if input_dataset == 1:
        name = 'Galaxies_400K'
    elif input_dataset == 2:
        name = 'Galaxies_1KK'
    elif input_dataset == 3: return

    dataset = Dataset(name)
    dataset.download()

def download_custom():
    print("Choose your TAP service:")
    print("\n\t---------- services -----------")
    print("\t\033[01;32m[1] <-> VOXastro\033[0m")
    print("\t\033[01;32m[2] <-> Simbad (University of Strasbourg)\033[0m")
    print("\t\033[01;32m[3] <-> NED (Caltech)\033[0m")
    print("\t--------------------------------\n")
    input_service = int(session.prompt(' > Your TAP service: ', auto_suggest=AutoSuggestFromHistory()))
    input_size = int(session.prompt(' > Size of dataset: ', auto_suggest=AutoSuggestFromHistory()))
    name = ''
    if input_service == 1:
        name = 'rcsed'
    elif input_service == 2:
        name = 'simbad'
    elif input_service == 3:
        name = 'ned'

    dataset = Dataset(name)
    dataset.download_via_tap(input_size)
    # dataset.add_new_dataset("custom", "https://custom.com")

def preconfiguration():
    print(f"\nSystem information: \033[01;32m{os.uname()}\033[0m")
    print("\nPreconfiguration:\n")
    prepared = session.prompt(' > Do you want to use prepared datasets? [y/n]: ', auto_suggest=AutoSuggestFromHistory())

    if prepared == "y":
        download_prepared_datasets()
    else:
        download_custom()

    print("\nChoose file with your data [.csv file]:")

    data_tables = os.walk('./data')

    print(data_tables)
    elem = None
    print("\n\t---------- data -----------")
    for _, _, elem in data_tables:
        for name in elem:
            print(f"\t\033[01;32m[{elem.index(name) + 1}] <-> {name}\033[0m")
    print("\t---------------------------\n")

    table_num = int(session.prompt(f" > Enter number of your table [1-{len(elem)}]: ", auto_suggest=AutoSuggestFromHistory()))
    path = os.path.abspath('.') + '/data/' + elem[table_num - 1]

    print(f"Loading data from \033[01;32m{path}\033[0m...")
    t = time.time()
    df = pd.read_csv(path)
    t = time.time() - t

    print(f"Loading done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\
    We have data about \033[01;32m{len(df)}\033[0m galaxies.\n")

    return df


def main():
    doc_key = session.prompt(' > Do you want to open documentation? [y/n]: ', auto_suggest=AutoSuggestFromHistory())
    if doc_key == 'y':
        print("\nOpening documentation...")
        url = 'https://galaxywitness.rtfd.io' #'file://' + os.path.abspath('.') + '/docs/build/html/index.html'
        webbrowser.open(url, new=2)
        print("Done\033[01;32m \u2714\033[0m")

    try:
        df = preconfiguration()
    except ValueError as e:
        raise Exception("\033[01;31mFolder 'data' does not exist or empty!\033[0m") from e

    n_gal = int(session.prompt(f" > Enter number of galaxies [10...{len(df)}]: ", auto_suggest=AutoSuggestFromHistory()))

    print("Choose type of complex:")
    print("\n\t---------- complexes -----------")
    print("\t\033[01;32m[1] <-> Vietoris-Rips complex\033[0m")
    print("\t\033[01;32m[2] <-> Alpha complex\033[0m")
    print("\t\033[01;32m[3] <-> Witness complex\033[0m")
    print("\t--------------------------------\n")
    type_of_complex = int(session.prompt(" > Your complex: ", auto_suggest=AutoSuggestFromHistory()))
    if type_of_complex == 1:
        type_of_complex = "rips"
    elif type_of_complex == 2:
        type_of_complex = "alpha"
    elif type_of_complex == 3:
        type_of_complex = "witness"

    if type_of_complex == "witness":
        n_landmarks = int(session.prompt(f" > Enter number of landmarks [10...{n_gal}]: ", auto_suggest=AutoSuggestFromHistory()))
    else:
        n_landmarks = n_gal

    # n_jobs = int(input("Enter number of processes: "))

    r_max = 7.5
    first_witness = 0
    tomato_key = 'y'
    column_names = []#['RAJ2000_gal', 'DEJ2000_gal', 'z_gal']
    isomap_eps = 0
    key_plot_cloud = 'y'
    key_anim = 'y'
    key_save = 'n'
    key_complex_type = 'gudhi'
    key_fig = 'mpl'
    # max_edge_length = np.inf
    sparse = None

    key_plot_cloud = session.prompt(' > Do you want plot the point cloud? [y/n]: ', auto_suggest=AutoSuggestFromHistory())
    key_anim = session.prompt(f" > Do you want watch the animation of {type_of_complex} filtration? [y/n]: ", auto_suggest=AutoSuggestFromHistory())
    if key_anim == 'y':
        key_fig = session.prompt(
            " > What will we use for the animation of a filtered complex? [plotly(more slow, but more cool)/mpl(matplotlib)]: ", auto_suggest=AutoSuggestFromHistory())
    key_save = session.prompt(" > Do you want save all plots to ./imgs? [y/n]: ", auto_suggest=AutoSuggestFromHistory())
    tomato_key = session.prompt(" > Do you want run tomato clustering? [y/n]: ", auto_suggest=AutoSuggestFromHistory())
    key_adv = session.prompt(" > Advanced configuration? [y/n]: ", auto_suggest=AutoSuggestFromHistory())
    if key_adv == 'y':
        key_complex_type = session.prompt(" > What type of simplicial complex will we use? [gudhi/custom]: ", auto_suggest=AutoSuggestFromHistory())
        r_max = float(session.prompt(
            " > Enter max value of filtration [usually \u2264 15, the more the slower calculations]: ", auto_suggest=AutoSuggestFromHistory()))
        if type_of_complex == "witness":
            first_witness = int(session.prompt(f" > Enter first witness [range: 0...{len(df) - n_gal}]: ", auto_suggest=AutoSuggestFromHistory()))
        if r_max == -1:
            r_max = None
        if type_of_complex == "rips":
            sparse = float(session.prompt(" > Enter 'sparse' parameter to build sparse Rips or -1: ", auto_suggest=AutoSuggestFromHistory()))
            if sparse == -1:
                sparse = None

        # if type_of_complex == 'witness':
        #     isomap_eps = float(input(" > Enter\033[01;32m isomap\033[0m parameter [0 - don't compute isomap metric]: "))

        # cosmology = input("Enter cosmology model: ")

    path_to_save = None
    time_list = list(time.localtime())
    time_str = ''
    for i in range(6):
        time_str += str(time_list[i])

    if key_save == 'y':
        path_to_save = os.path.abspath('.') + '/imgs/' + time_str + f"-{n_gal}-{n_landmarks}"
        if not os.path.isdir('imgs'):
            os.mkdir('imgs')
        os.mkdir(path_to_save)

    section()
    print(f"Info about the handled table: \n\033[01;32m{df.info}\033[0m\n")

    list_names = list(df)
    pause()
    print("\nChoosing names of 3 columns for right ascension [ra], declination [dec] and redshift [z]:")

    print("\n\t---------- column names -----------")
    for elem in list_names:
        if list_names.index(elem) != 0:
            print(f"\t\033[01;32m[{list_names.index(elem)}] <-> {elem}\033[0m")

    print("\t-----------------------------------\n")

    column_nums = []

    for name in ('ra', 'dec', 'z'):
        column_nums.append(
            int(session.prompt(f" > Choose column for {name}, from list above (column names): ", auto_suggest=AutoSuggestFromHistory())))

    column_names = [list(df)[column_nums[0]], list(df)[column_nums[1]], list(df)[column_nums[2]]]

    section()

    print("Preprocessing data and plot the point cloud...")

    t = time.time()

    points = np.array(df[column_names].values[first_witness:n_gal + first_witness])

    coord = SkyCoord(
        ra=points[:, 0] * u.degree,
        dec=points[:, 1] * u.degree,
        distance=Distance(z=points[:, 2])
    )

    points = np.transpose(np.array(coord.cartesian.xyz), (1, 0))
    landmarks = np.zeros((n_landmarks, 3))

    if type_of_complex == "witness":
        witnesses = points

        landmarks_factor = int(n_gal / n_landmarks)
        landmarks_idxs = np.zeros(n_landmarks, dtype=int)

        for i, j in zip(range(0, n_gal, landmarks_factor), range(n_landmarks)):
            landmarks[j, :] = witnesses[i, :]
            landmarks_idxs[j] = i
    else:
        landmarks = points

    t = time.time() - t

    print(
        f"\033[FPreprocessing data and plot the point cloud... done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.")

    pause()
    section()

    if key_plot_cloud == 'y':
        print("Trying plot data cloud...")
        if type_of_complex == "witness":
            plot_data_cloud(witnesses, landmarks, key_save, path_to_save)
        else:
            plot_data_cloud_alpha(landmarks, key_save, path_to_save)
        print("\033[FTrying plot data cloud... done \033[01;32m\u2714\033[0m")
        pause()
        section()

    print(f"Computing persistence with {type_of_complex} filtration...")

    t = time.time()
    complex_ = BaseComplex()
    if type_of_complex == "witness":
        complex_.__class__ = WitnessComplex
        complex_.__init__(landmarks, witnesses, landmarks_idxs, isomap_eps=isomap_eps)
    elif type_of_complex == "alpha":
        complex_.__class__ = AlphaComplex
        complex_.__init__(points=landmarks)
    else:
        complex_.__class__ = RipsComplex
        complex_.__init__(points=landmarks, max_edge_length=r_max, sparse=sparse)

    complex_.compute_simplicial_complex(d_max=MAX_DIM, r_max=r_max, custom=(key_complex_type == 'custom'))

    if type_of_complex == 'alpha':
        magnitude_levels = [(r_max ** 2),  (r_max ** 2) / 2.0, (r_max ** 2) / 2.0]
    else:
        magnitude_levels = [r_max, r_max / 2.0,  r_max / 2.0]

    simplex_tree = complex_.simplex_tree

    betti = complex_.get_persistence_betti(dim=MAX_DIM, magnitudes=magnitude_levels)

    t = time.time() - t

    print(f"\033[FComputing persistence with {type_of_complex} filtration... done\033[01;32m \u2714\033[0m in \033[01;32m{t}\033[0m sec.\n")
    pause()

    print("The \033[01;32msimplex tree\033[0m constructed \033[01;32m\u2714\033[0m")
    print(f"\t\033[01;32msimplex tree stats:\n\t dim: {simplex_tree.dimension()}\033[0m")
    print(f"\t\033[01;32m number of vertices (landmarks): {simplex_tree.num_vertices()}\033[0m")
    print(f"\t\033[01;32m total number of simplices: {simplex_tree.num_simplices()}\033[0m")
    print(f"\t\033[01;32m persistence betti numbers: {betti}\033[0m")

    section()

    print("\nDrawing persistence diagram and barcode...")

    draw_diagrams_and_animation(complex_, key_anim, path_to_save, key_fig)

    print("\033[F\033[F\033[FDrawing persistence diagram and barcode... done \033[01;32m \u2714\033[0m")
    pause()

    section()

    if tomato_key == 'y':
        if type_of_complex != "witness":
            # clustering(complex_, complex_.witnesses, path_to_save, den_type, graph_type_)
            tomato = clustering(complex_, complex_.points, r_max, path_to_save)

            print("Warning: results down below is additional experimental functionality")
            cl_1, cl_2 = Clusterization(points), Clusterization(points)
            cl_1.import_clustering(tomato.labels_)
            cl_2.tomato(r_max)
            print("Draw clustering...")
            cl_2.draw_clustering()
            print("Draw projections of clustering on random planes...")
            cl_2.draw_projections(2)
            diff = cl_1.compare_clusterization(cl_2)
            print(f"Test Difference between clusterings is \033[01;32m {diff}\033[0m")



if __name__ == '__main__':
    print("\033[FLoading... done \033[01;32m\u2714\033[0m")
    pause()
    main()
