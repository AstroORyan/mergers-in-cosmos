import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u

import sys
import logging
import os
import glob
from tqdm import tqdm

def calc_sep(ra1, dec1, ra2, dec2, z_1, z_2, cosmo):
    
    d1 = cosmo.comoving_distance(z_1).to(u.kpc)
    d2 = cosmo.comoving_distance(z_1).to(u.kpc)
    
    c1 = SkyCoord(ra = ra1 * u.deg, dec = dec1 * u.deg, frame = 'fk5')
    c2 = SkyCoord(ra = ra2 * u.deg, dec = dec2 * u.deg, frame = 'fk5')
    
    ang_sep = c1.separation(c2).to(u.arcmin)
    conversion = cosmo.kpc_proper_per_arcmin(z_1)
    
    proj_sep = ang_sep * conversion
    
    return float(proj_sep.to(u.Mpc) / u.Mpc)

def get_n_neighbours(gal_id, ra, dec):
    
    end_index = 0
    ang = 0.1
    while end_index < 5:
        record = data[(data['ALPHA_J2000'] < (ra + ang)) & (data['ALPHA_J2000'] > (ra - ang)) & (data['DELTA_J2000'] < (dec + ang)) & (data['DELTA_J2000'] > (dec - ang))]

        table = Table(record)

        prim_galaxy_record = record[record['ID'] == gal_id]
        prim_z = prim_galaxy_record['ez_z_phot'][0]

        df = table.to_pandas()[['ID','ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'lp_type']]
        df = df.query('lp_type == 0').drop(columns = 'lp_type').dropna()
        df = df.query('ID != @gal_id')

        if len(df) < 0.5:
            return 'null'

        df_z_diff = (
            df
            .assign(z_diff = df.ez_z_phot.apply(lambda x: abs(x - prim_z)))
        )

        df_z_cut = df_z_diff.query('z_diff < 0.005')
        if len(df_z_cut) < 1:
            return {'IDs': [], 'separations': [], 'N_1' : None, 'N_2' : None, 'N_3' : None, 'N_4': None, 'N_5' : None}

        df_sep = (
            df_z_cut
            .assign(separation = df_z_cut.apply(lambda row: calc_sep(ra, dec, row.ALPHA_J2000, row.DELTA_J2000, prim_z, row.ez_z_phot, cosmo), axis = 1))
        )

        df_sorted = df_sep.sort_values(by = 'separation', ascending = True)

        end_index = 5
        if len(df_sorted) < end_index:
            logging.info('Expanding Search Range...')
            end_index = len(df_sorted)
            ang += 0.05

    df_nearest = df_sorted[:end_index]

    nearest = {'IDs': [], 'separations': [], 'N_1' : None, 'N_2' : None, 'N_3' : None, 'N_4': None, 'N_5' : None}
    for i in range(end_index):
        nearest['IDs'].append(df_nearest.ID.iloc[i])
        nearest['separations'].append(df_nearest.separation.iloc[i])
        nearest[f'N_{i+1}'] = i+1 / (np.pi * (df_nearest.separation.iloc[i])**2)
        
    return nearest

def main():
    global data
    data_folder = '/mmfs1/storage/users/oryan/cosmos-data'
    cosmos_folder = '/mmfs1/home/users/oryan/cosmos/data'
    results = f'/mmfs1/storage/users/oryan/cosmos/results'

    with fits.open(f'{drive_folder}/COSMOS2020_CLASSIC_R1_v2.1_p3.fits.gz') as hdul:
        header = hdul[1].header
        data = hdul[1].data

    data = Table(data)['ID', 'ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'lp_type']

    df = pd.read_csv(f'{data_folder}/catalogue-matched-cosmos-2020.csv', index_col = 0)

    df_red = (
        df[['SourceID', 'ID_1', 'ALPHA_J2000_1', 'DELTA_J2000_1']]
    )

    coord_dict = df_red.set_index('SourceID').to_dict(orient = 'index')

    global cosmo
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    results_dict = {}
    counter = 0
    mult = 1
    done_list = []
    for sourceid, values in tqdm(coord_dict.items()):
        if sourceid in done_list:
            continue
        results_dict[sourceid] = get_n_neighbours(values['ID_1'], values['ALPHA_J2000_1'], values['DELTA_J2000_1'])
        counter += 1
        
        if counter / 50 == 10:
            df_Ns = pd.DataFrame.from_dict(results_dict, orient = 'index').reset_index()
            df_Ns.to_csv(f'{results}/tmp-nearest-neighbours-corr-sample-{mult*counter}.csv')
            counter = 0
            mult += 1
            del df_Ns
            results_dict = {}

    df_Ns = pd.DataFrame.from_dict(results_dict, orient = 'index').reset_index()

    csv_files = glob.glob(f'{results}/tmp-nearest-neighbours-corr-sample-*.csv')
    for i in csv_files:
        df_tmp = pd.read_csv(i, index_col = 0)
        df_Ns = pd.concat([df_Ns, df_tmp])  

    df_Ns.to_csv(f'{results}/nearest-neighbours-corr-sample.csv')

    for i in csv_files:
        os.remove(i)

drive_folder = 'E:/cosmos-data'

if __name__ == '__main__':

    main()