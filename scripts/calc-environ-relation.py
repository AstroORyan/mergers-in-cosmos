import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u

import multiprocessing

import os
import logging
import time
import datetime
import glob
from tqdm import tqdm

def calc_sep(ra1, dec1, ra2, dec2, z_1, z_2, cosmo):
        
    c1 = SkyCoord(ra = ra1 * u.deg, dec = dec1 * u.deg, frame = 'fk5')
    c2 = SkyCoord(ra = ra2 * u.deg, dec = dec2 * u.deg, frame = 'fk5')
    
    ang_sep = c1.separation(c2).to(u.arcmin)
    conversion = cosmo.kpc_proper_per_arcmin(z_1)
        
    proj_sep = ang_sep * conversion
    
    return float(proj_sep.to(u.Mpc) / u.Mpc)

def get_n_neighbours(gal_id, ra, dec):

    nearest = {'IDs': [], 'separations': [], 'N_1' : np.nan, 'N_2' : np.nan, 'N_3' : np.nan, 'N_4': np.nan, 'N_5' : np.nan}

    # if gal_id % 500 == 0:
    #     logging.info(gal_id)

    # record = data[(data['ALPHA_J2000'] < (ra + ang)) & (data['ALPHA_J2000'] > (ra - ang)) & (data['DELTA_J2000'] < (dec + ang)) & (data['DELTA_J2000'] > (dec - ang))]

    table = Table(data)

    prim_galaxy_record = data[data['ID'] == gal_id]
    prim_z = prim_galaxy_record['ez_z_phot'][0]

    df = table.to_pandas()[['ID','ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'lp_type']]
    df = df.query('lp_type == 0').drop(columns = 'lp_type').dropna()
    df = df.query('ID != @gal_id')

    if len(df) < 0.5:
        return nearest

    df_z_diff = (
        df
        .assign(z_diff = df.ez_z_phot.apply(lambda x: abs(x - prim_z)))
    )

    df_z_cut = df_z_diff.query('z_diff < 0.005')
    if len(df_z_cut) < 1:
        return {'IDs': [], 'separations': [], 'N_1' : np.nan, 'N_2' : np.nan, 'N_3' : np.nan, 'N_4': np.nan, 'N_5' : np.nan}

    df_sep = (
        df_z_cut
        .assign(separation = df_z_cut.apply(lambda row: calc_sep(ra, dec, row.ALPHA_J2000, row.DELTA_J2000, prim_z, row.ez_z_phot, cosmo), axis = 1))
    )

    df_sorted = df_sep.sort_values(by = 'separation', ascending = True)

    end_index = 5
    df_nearest = df_sorted[:end_index]

    if len(df_nearest) < 5:
        return nearest

    for i in range(end_index):
        nearest['IDs'].append(df_nearest.ID.iloc[i])
        nearest['separations'].append(df_nearest.separation.iloc[i])
        nearest[f'N_{i+1}'] = i+1 / (np.pi * (df_nearest.separation.iloc[i])**2)
    
    pbar.update(1)
            
    return nearest

def assign_values(row):
    sourceid = row[0]
    ra = row[1]['ALPHA_J2000']
    dec = row[1]['DELTA_J2000']

    results_dict = {}
    if len(row) < 0.5:
        return None
    results_dict[sourceid] = get_n_neighbours(sourceid, ra, dec)
    
    return results_dict

def main():
    global data, cosmo, pbar

    data_folder = '/mmfs1/storage/users/oryan/cosmos-data'
    results = f'/mmfs1/storage/users/oryan/cosmos/results'

    logging.info('Loading in COSMOS data...')
    with fits.open(f'{data_folder}/COSMOS2020_CLASSIC_R1_v2.1_p3.fits.gz') as hdul:
        data = hdul[1].data
    logging.info('Completed.')

    data = data[data['lp_type'] == 0]

    data_tab = Table(data)

    del data
    data = data_tab['ID', 'ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot', 'lp_type']
    del data_tab
    time.sleep(0.5)
    data_df = data.to_pandas().dropna()
    data_dict = data_df.set_index('ID').to_dict(orient = 'index')
    del data_df
    time.sleep(0.5)

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    
    logging.info('Beginning to calculate distance matrices...')

    t1 = datetime.datetime.now()
    pbar = tqdm(total=len(list(data_dict.items()))/4)
    with multiprocessing.Pool(processes = 4) as pool:
        res = pool.map(assign_values, list(data_dict.items()))

    t2 = datetime.datetime.now()
    logging.info(f'Running on {len(data_dict)} entries was {t2 - t1}')   

        # res = [p.get() for p in processes]

    # with Parallel(n_jobs = 8, backend = 'multiprocessing') as parallel:
    #     res = parallel(delayed(assign_values)(list(data_dict.items())[i]) for i in tqdm(range(len(data_dict))))    

    results_dict = {}

    for i in res:
        key = list(i.keys())[0]
        results_dict[key] = i[key]

    del data, cosmo
    logging.info('Complete.')
    
    df_Ns = pd.DataFrame.from_dict(results_dict, orient = 'index').reset_index()
        
    df_Ns.to_csv(f'{results}/nearest-neighbours-entire.csv')


if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO, format = '%(asctime)s: %(message)s', filename = '/mmfs1/home/users/oryan/calc-environ.log')
    logging.info(f'number of CPUs: {os.cpu_count()}')

    main()