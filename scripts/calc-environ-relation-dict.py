## Imports
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u

import logging
import os
import glob
from tqdm import tqdm
tqdm.pandas()

## Functions
def calc_sep(ra1, dec1, ra2, dec2, conversion):
    
    c1 = SkyCoord(ra = ra1 * u.deg, dec = dec1 * u.deg, frame = 'fk5')
    c2 = SkyCoord(ra = ra2 * u.deg, dec = dec2 * u.deg, frame = 'fk5')
    
    ang_sep = c1.separation(c2).to(u.arcmin)
        
    proj_sep = ang_sep * conversion
    
    return float(proj_sep.to(u.Mpc) / u.Mpc)

## Main Function
def main():
    data_folder = '/mmfs1/storage/users/oryan/cosmos-data'
    results = f'/mmfs1/storage/users/oryan/cosmos/results'

    logging.info('Importing COSMOS data...')
    with fits.open(f'{data_folder}/COSMOS2020_CLASSIC_R1_v2.1_p3.fits.gz') as hdul:
        data_rec = hdul[1].data
    logging.info('Completed.')

    logging.info('Reducing Data Table...')
    data_rec = data_rec[data_rec['lp_type'] == 0]
    data_rec = Table(data_rec)
    data = data_rec['ID', 'ALPHA_J2000', 'DELTA_J2000', 'ez_z_phot']
    del data_rec
    logging.info('Completed.')

    logging.info('Setting up loop...')
    ids = list(data['ID'])
    ang = 0.1
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
    results_dict = {}
    counter = 0

    logging.info('Beginning loop...')
    for i in tqdm(ids):
        tmp_dict = {'IDs': [], 'N_1': np.nan, 'N_2': np.nan, 'N_3': np.nan, 'N_4': np.nan, 'N_5': np.nan}
        entry = data[data['ID'] == i]
        if np.isnan(entry['ALPHA_J2000']) or np.isnan(entry['DELTA_J2000']) or np.isnan(entry['ez_z_phot']):
            continue
        
        ra = entry['ALPHA_J2000']
        dec = entry['DELTA_J2000']
        z_phot = entry['ez_z_phot']
        conversion = cosmo.kpc_proper_per_arcmin(z_phot)

        
        record = data[(data['ALPHA_J2000'] < (ra + ang)) & (data['ALPHA_J2000'] > (ra - ang)) & (data['DELTA_J2000'] < (dec + ang)) & (data['DELTA_J2000'] > (dec - ang))]
        record = record[record['ID'] != i]
        
        record = record[(record['ez_z_phot'] > z_phot - 0.005) & (record['ez_z_phot'] < z_phot + 0.005)]
        
        record_df = record.to_pandas()
        
        if len(record_df) == 0.0:
            results_dict[i] = tmp_dict
            continue
        
        record_df = (
            record_df
            .assign(seperations = record_df.apply(lambda row: calc_sep(ra, dec, row.ALPHA_J2000, row.DELTA_J2000, conversion), axis = 1))
        )
        
        record_df = record_df.sort_values('seperations', ascending = True)
        
        record_df = record_df[:5][['ID', 'seperations']]
            
        for j in range(len(record_df)):
            tmp_dict['IDs'].append(record_df.ID.iloc[j])
            tmp_dict[f'N_{j+1}'] = j+1 / (np.pi * (record_df.seperations.iloc[j])**2)
            
        results_dict[i] = tmp_dict
        
        if len(results_dict) > 50000:
            results_df = pd.DataFrame.from_dict(results_dict, orient = 'index').reset_index().rename(columns = {'index' : 'ID'})
            
            results_df.to_csv(f'{results}/full-sample-{counter * 50}-{(counter + 1) * 50}.csv')
            
            counter += 1
            results_dict = {}
            del results_df
    
    logging.info('Completed loop.')
    
    logging.info('Combining constituent files...')
    csv_files = glob.glob(f'{results}/full-sample-*-*.csv')

    for counter, i in enumerate(csv_files):
        if counter == 0:
            df = pd.read_csv(i, index_col = 0)
            continue
        df_tmp = pd.read_csv(i, index_col = 0)
        df = pd.concat([df, df_tmp])
    
    logging.info('Completed.')
        
    df.to_csv(f'{results}/full-sample-all.csv')

    logging.info('Algorithm Completed.')



## Initialisation
if __name__ == '__main__':

    logging.basicConfig(level = logging.INFO, format = '%(asctime)s: %(message)s', filename = '/mmfs1/home/users/oryan/calc-environ.log')
    logging.info(f'number of CPUs: {os.cpu_count()}')

    main()