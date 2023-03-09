import os
import csv
import requests
from tqdm import tqdm
import pyvo as vo


class Dataset:
    """
    Class to handle prepared datasets

    :param name: name of dataset
    :type name: str

    """
    inner_names = {"Galaxies_400K": "Galaxies_400K.csv",
                   "Galaxies_1KK": "Galaxies_1KK.csv",
                   "rcsed": "rcsed.csv",
                   "simbad": "simbad.csv",
                   "ned": "ned.csv"}

    addresses = {"Galaxies_400K":
            "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_glist_s.csv",
                 "Galaxies_1KK":
            "https://raw.githubusercontent.com/Arrrtemiron/galaxy_witness_datasets/main/result_rcsed_vo.csv",
                 "rcsed": "http://rcsed-vo.sai.msu.ru/tap/",
                 "simbad": "http://simbad.u-strasbg.fr/simbad/sim-tap/",
                 "ned": "http://ned.ipac.caltech.edu/tap/"}

    def __init__(self, name: str):
        self.name = name
        self.url = ''
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
            unit_divisor=chunk_size,
        ) as bar_:
            for data in resp.iter_content(chunk_size=chunk_size):
                bar_.update(file.write(data))
        
        os.chdir("..")

    def download_via_tap(self, size: int = 100000) -> None:
        """
        Download current prepared dataset via TAP

        :param size: size of dataset
        :type size: int
        """
        assert self.dataset_prepared
        tap_service = vo.dal.TAPService(self.url)
        tap_service.describe()
        oid, redshift, table, otype = '', '', '', ''
        if self.name == "rcsed": oid = "objid"; redshift = "z"; table = "specphot.rcsed"
        elif self.name == "simbad": oid = "main_id"; redshift = "rvz_redshift"; table = "basic"; otype = "AND otype = 'galaxy..'"
        elif self.name == "ned": oid = "prefname"; redshift = "z"; table = "objdir"; otype = "AND (pretype = 'G' OR pretype = 'QSO')"

        tap_results = tap_service.run_async(f"SELECT {oid}, ra, dec, {redshift} FROM {table} WHERE ra is not NULL AND \
                                            dec is not NULL AND {redshift} > 0 {otype} ORDER BY {redshift}", maxrec = size)
        
        header = [oid, 'ra', 'dec', redshift]
        rows = []

        for i in range(size):
            cur = []
            for j in header:
                cur.append(tap_results[i][j])

            rows.append(cur)

        if not os.path.isdir('data'):
            os.mkdir('data')
        os.chdir("./data")

        with open(self.inner_names[self.name], 'w', encoding='UTF8', newline='') as f:
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
        print("New dataset added successfully:", name, "from", url)
        print("You can change current dataset to this one by calling change_dataset_to method")
        

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
