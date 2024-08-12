import pandas as pd
from astropy.io import fits
from astropy.table import QTable
import numpy as np

y = fits.open('/project/chihway/dhayaa/DECADE/Imsim_Inputs/DESY3_Deepfields_V2BalrogCuts.fits')[1].data
y = QTable(y)
x = pd.read_csv('/project/chihway/dhayaa/DECADE/Imsim_Inputs/deepfields_raw_with_redshifts.csv.gz')

uniquee, inds_y, inds_X = np.intersect1d(y['ID'], x['ID'].values, return_indices = True)

y['FLUX_G']    = x['BDF_FLUX_DERED_CALIB_G'].values[inds_X][np.argsort(inds_y)]
y['FLUXERR_G'] = x['BDF_FLUX_ERR_DERED_CALIB_G'].values[inds_X][np.argsort(inds_y)]

assert np.allclose(y['ID'], x['ID'].values[inds_X][np.argsort(inds_y)]), "Matching has failed...."

fits.writeto('/project/chihway/dhayaa/DECADE/Peter_Files/DESY3_Deepfields.fits', y.as_array(), overwrite = True)