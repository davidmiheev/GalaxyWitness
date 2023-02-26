import os
import requests
from tqdm import tqdm
import pyvo as vo
import csv

class Dataset:
    """
    Class to handle prepared datasets

    :param name: name of dataset
    :type name: str

    """
    inner_names = {"Galaxies_400K": "Galaxies_400K.csv",
                   "Galaxies_1KK": "Galaxies_1KK.csv"}
    addresses = {"Galaxies_400K":
            "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_glist_s.csv",
                 "Galaxies_1KK":
            "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_rcsed_vo.csv"}

    def __init__(self, name: str):
        self.name = name
        if name in self.addresses:
            self.url = self.addresses[name]
            self.dataset_prepared = True
        else:
            print("Incorrect name of dataset")
            self.dataset_prepared = False


    def download(self, chunk_size=1024) -> None:
        """
        Download current prepared dataset

        """
        assert self.dataset_prepared
        if not os.path.isdir('data'):
            os.mkdir('data')
        os.chdir("./data")
        resp = requests.get(self.url, stream=True, timeout=60)
        total = int(resp.headers.get('content-length', 0))
        with open(self.inner_names[self.name], 'wb') as file, tqdm(
            desc=self.name,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar_:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar_.update(size)
        os.chdir("..")

    def download_pyvo():
        tap_service_1 = vo.dal.TAPService("http://rcsed-vo.sai.msu.ru/tap/")
        tap_results_1 = tap_service_1.search("SELECT TOP 100000 OBJID, RA, DEC, Z FROM specphot.rcsed WHERE RA is not NULL AND \
                                             DEC is not NULL AND Z is not NULL", maxrec = 100000)
        
        header = ['objid', 'ra', 'dec', 'z']
        rows = []

        for i in range(100000):
            cur = []
            for j in header:
                cur.append(tap_results_1[i][j])

            rows.append(cur)

        if not os.path.isdir('data'):
            os.mkdir('data')
        os.chdir("./data")

        with open('test.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            writer.writerows(rows)

        os.chdir("..")


    def add_new_dataset(self, name: str, url: str) -> None:
        """
        Add new dataset by name and by URL where it can be retrieved

        :param name: name of dataset
        :type name: str
        :param name: URL
        :type name: str
        """
        self.inner_names[name] = name + '.csv'
        self.addresses[name] = url

    def change_dataset_to(self, name: str) -> None:
        """
        Change current dataset to another by name

        :param name: name of dataset
        :type name: str

        """
        if name in self.addresses:
            self.name = name
            self.url = self.addresses[name]
