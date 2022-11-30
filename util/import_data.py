import csv
import numpy as np

data_dir = '../data/'
geo_data_label = ['Di',	'Df', 'Dif', 'rho',	'VSA',	'GSA',	'VPOV',	'GPOV',	'POAV_vol_frac', 'PONAV_vol_frac',	'GPOAV', 'GPONAV',	
                  'POAV',	'PONAV', 'VF', 'VSA', 'GSA', 'PLD', 'LCD', 'LCD/PLD', 'UCV', 'GPV', 'density']

def import_data(filename):
    with open(data_dir + filename, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        data = []
        for line in rdr:
            data.append(line)
    data = np.array(data)
    n = data.shape[0]

    bulk_idx = np.where(data[0]=='bulk')[0][0]
    shear_idx = np.where(data[0]=='shear')[0][0]
    mask = np.where(np.isin(data[0], geo_data_label)==True)[0]
    
    data_x = data[1:n, mask].astype(float)
    bulk_y = data[1:n, bulk_idx].astype(float)
    shear_y = data[1:n, shear_idx].astype(float)

    print('='*30)
    print(f'Import data: {filename}')
    print(f'Geometry data shape: {data_x.shape}')
    print(f'Bulk data shape: {bulk_y.shape}')
    print(f'Shear data shape: {shear_y.shape}')
    print('='*30)

    return data_x, bulk_y, shear_y