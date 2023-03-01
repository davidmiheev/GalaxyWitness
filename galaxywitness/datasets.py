import os
import requests
from tqdm import tqdm

class Dataset:
    """
    Class to handle prepared datasets
    
    :param name: name of dataset
    :type name: str
    
    """
    inner_names = {"Galaxies_400K": "Galaxies_400K.csv"}
    addresses = {"Galaxies_400K": 
            "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_glist_s.csv"}
    
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
        self.name = name
        if name in self.addresses:
            self.url = self.addresses[name]
            self.dataset_prepared = True
        else:
            print("Incorrect name of dataset")
            self.dataset_prepared = False
