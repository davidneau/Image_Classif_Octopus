from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

monRepertoire = "./test/"
fichiers = [f for f in listdir(monRepertoire) if isfile(join(monRepertoire, f))]

df = pd.DataFrame(columns = ['id', 'label'])

iter = 0
for i in fichiers:
    df.loc[iter, 'id'] = i
    if "PTO" in i:
        df.loc[iter, 'label'] = 1
    else:
        df.loc[iter, 'label'] = 0
    iter += 1

df.to_csv('./labelstest.csv')
