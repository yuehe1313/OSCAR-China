
###############################################################################
# run OSCAR_China 1800-2018
###############################################################################

## imports 
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from core_fct.fct_genMC import generate_config
from core_fct.mod_process import OSCAR_landC 
from core_fct.fct_process_alt import full_LUC

OSCAR_China = full_LUC(OSCAR_landC)

## filepath
file_pathin = '/home/heyue/OSCAR_China/'

## set run options
ind0, ind1 = 1800, 2018
nMC = 10

n_run = 1

for n_run in range(1,101):

    ## load primary parameters 
    Par = xr.open_dataset(file_pathin + 'input_data/land_TRENDYv7_China.nc')
    Par = generate_config(Par, nMC)

    ## atmospheric CO2 
    For_co2 = xr.open_dataset(file_pathin + 'input_data/global_co2_ann_1700_2018.nc')
    
    ## climate 
    For_clim = xr.open_dataset(file_pathin + 'input_data/land-climate_China_CRU-TS.nc')

    ## LUC 
    For_luc = xr.open_dataset(file_pathin + f'input_data/land-use_China_1800_2018.nc')
    For_luc = For_luc.sel(bio_land = ['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban']).rename({'d_Agross':'d_Acover'}).drop('d_Anet')

    ## format drivers
    For = xr.merge([For_luc, For_clim, For_co2]).sel(year=slice(ind0, ind1))
    For['D_CO2'] = For.CO2 - Par.CO2_0
    For['D_Tl'] = For.Tl - For.Tl.sel(year=slice(1901, 1920)).mean('year')
    For['D_Pl'] = For.Pl - For.Pl.sel(year=slice(1901, 1920)).mean('year')
    For = For.drop(['CO2', 'Tl', 'Pl'])
    For = For.fillna(0).sel(year=slice(1900)).combine_first(For)

    ## swap preindustrial land-cover
    Par['Aland_0'] = For.Aland_0
    For = For.drop('Aland_0')

    ## run model 
    Out_full = OSCAR_China(Ini=None, Par=Par, For=For, adapt_nt=False)

    # save data
    Out_full.to_netcdf(file_pathin + f'results/n{n_run}_test_grid_full_Out.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Out_full})
    Par.to_netcdf(file_pathin + f'results/n{n_run}_test_grid_full_Par.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Par})
    if n_run==1: For.to_netcdf(file_pathin + f'results/n{n_run}_test_grid_full_For.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in For})
    
    del Par, For, Out_full


###############################################################################
# postprocess 1900-2018 
###############################################################################

## imports 
import os
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
from core_fct.mod_process import OSCAR_landC 
from core_fct.fct_process_alt import full_LUC

OSCAR_China = full_LUC(OSCAR_landC)

## filepath
file_pathin = '/home/heyue/OSCAR_China/'

## managed forest area
data = xr.open_dataset(file_pathin + f'/input_data/NonIntactForestArea_1900_2100.nc')
NonIntactForestArea = data['Continued'].sel(year=slice(1900,2018))

n_run = 1

## loops
for n_run in range(1,101):

    print(n_run)

    ## load files
    Par = xr.open_dataset(file_pathin + f'results/n{n_run}_test_grid_full_Par.nc', chunks={'config':5})
    For = xr.open_dataset(file_pathin + f'results/n1_test_grid_full_For.nc', chunks={'config':5})
    Out = xr.open_dataset(file_pathin + f'results/n{n_run}_test_grid_full_Out.nc', chunks={'config':5})


    ## function to get any variable
    Get = lambda var: OSCAR_China[var](Out, Par, For, recursive=True, time_axis='year')

    ## ELUC (direct)
    ELUC_pattern = -Get('D_NBP_bk').sum('bio_to', min_count=1).sum('bio_from', min_count=1)
    ELUC_sum = ELUC_pattern.sum('reg_land', min_count=1)

    ELUC_Cveg_sum = (Get('D_Egraz_bk') + Get('D_Eharv_bk') + Get('D_Efire_bk') + Get('D_Fmort1_bk') + Get('D_Fmort2_bk') - Get('D_NPP_bk')).sum('bio_to', min_count=1).sum('bio_from', min_count=1).sum('reg_land')
    ELUC_Csoil_sum = (Get('D_Rh1_bk') + Get('D_Rh2_bk') - Get('D_Fmort1_bk') - Get('D_Fmort2_bk')).sum('bio_to', min_count=1).sum('bio_from', min_count=1).sum('reg_land')
    ELUC_Chwp_sum = Get('D_Ehwp').sum('box_hwp', min_count=1).sum('bio_to', min_count=1).sum('bio_from', min_count=1).sum('reg_land')

    ## LASC 
    D_fveg =  Get('D_eharv') + Get('D_efire') + Get('D_egraz') + Get('D_fmort1') + Get('D_fmort2') - Get('D_npp') + (Par.igni_0 + Par.harv_0 + Par.graz_0 + Par.mu1_0 + Par.mu2_0) * Get('cveg_0') - Par.npp_0
    LASC_Cveg = (D_fveg.rename({'bio_land':'bio_to'}) - D_fveg.rename({'bio_land':'bio_from'})) * For.d_Acover.cumsum('year')
    D_fsoil = Get('D_rh1') + Get('D_rh2') - Get('D_fmort1') - Get('D_fmort2') + Par.rho1_0 * Get('csoil1_0') + Par.rho2_0 * Get('csoil2_0') - (Par.mu1_0 + Par.mu2_0) * Get('cveg_0')
    LASC_Csoil = (D_fsoil.rename({'bio_land':'bio_to'}) - D_fsoil.rename({'bio_land':'bio_from'})) * For.d_Acover.cumsum('year')

    LASC_Cveg_sum =  LASC_Cveg.sum('bio_to', min_count=1).sum('bio_from', min_count=1).sum('reg_land')
    LASC_Csoil_sum = LASC_Csoil.sum('bio_to', min_count=1).sum('bio_from', min_count=1).sum('reg_land')
    
    LASC_sum = LASC_Cveg_sum + LASC_Csoil_sum 

    ## LSNK 
    LSNK_Cveg_sum = (D_fveg * (Par.Aland_0 + Out.D_Aland)).sum('reg_land').sum('bio_land')
    LSNK_Csoil_sum = (D_fsoil * (Par.Aland_0 + Out.D_Aland)).sum('reg_land').sum('bio_land')
    LSNK_sum = LSNK_Cveg_sum + LSNK_Csoil_sum 

    ## LSNKman (indirect)
    D_nbp = -1*Get('D_nbp')
    LSNKman_sum = (D_nbp.sel(bio_land='Forest') * NonIntactForestArea).sum(dim=['reg_land'])
    LSNKman_pattern = (D_nbp.sel(bio_land='Forest') * NonIntactForestArea)

    ## FNETman (total)
    FNETman_sum = LSNKman_sum + ELUC_sum
    FNETman_pattern = LSNKman_pattern + ELUC_pattern

    ## FNET
    FNET_sum = LSNK_sum + ELUC_sum

    ## broken-down by LUC type
    b2b_grouping = xr.DataArray(np.array([[2, 0, 0, 0, 0], [2, 2, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 1, 1], [2, 1, 1, 1, 1]]), 
        coords={'bio_from':['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban'], 'bio_to':['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban']}, dims=['bio_from', 'bio_to'])
    b2b_names = ['Deforestation', 'Other land-use transitions', 'Forestation and Wood harvest']           

    ELUC_b2b_all = -1*Get('D_NBP_bk')
    ELUC_b2b_all.coords['b2b'] = b2b_grouping

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ELUC_b2b = ELUC_b2b_all.groupby('b2b').sum('stacked_bio_from_bio_to')

    ELUC_b2b_pattern = ELUC_b2b.transpose('year', 'b2b','reg_land','config')
    ELUC_b2b_sum = ELUC_b2b.sum('reg_land')

    Cdensity = (Get('cveg_0') + Get('D_cveg') + Get('csoil1_0') + Get('D_csoil1') + Get('csoil2_0') + Get('D_csoil2')).sel(bio_land='Forest') 


    ## save to postprocess files
    ALL = xr.Dataset({'D_nbp':D_nbp, 'Cdensity': Cdensity,
                        'ELUC_sum':ELUC_sum,'ELUC_pattern':ELUC_pattern,'ELUC_Chwp_sum':ELUC_Chwp_sum,'ELUC_Csoil_sum':ELUC_Csoil_sum,'ELUC_Cveg_sum':ELUC_Cveg_sum,
                        'ELUC_b2b_sum':ELUC_b2b_sum,'ELUC_b2b_pattern':ELUC_b2b_pattern,
                        'LASC_sum':LASC_sum,'LASC_Cveg_sum':LASC_Cveg_sum, 'LASC_Csoil_sum':LASC_Csoil_sum,
                        'LSNK_sum':LSNK_sum,'LSNK_Cveg_sum':LSNK_Cveg_sum, 'LSNK_Csoil_sum':LSNK_Csoil_sum,
                        'LSNKman_sum':LSNKman_sum,'LSNKman_pattern':LSNKman_pattern,
                        'FNETman_sum':FNETman_sum,'FNETman_pattern':FNETman_pattern,
                        'FNET_sum':FNET_sum})

    ALL = ALL.sel(year=slice(1900,2018))
    ALL = ALL.transpose(..., 'config')
                        
    ALL.to_netcdf(file_pathin + f'results/postprocess/n{n_run}_test_grid_full_Out_postprocess.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in ALL})


###############################################################################
# weights 
###############################################################################

## imports 
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from core_fct.mod_process import OSCAR_landC 
from core_fct.fct_process_alt import full_LUC

OSCAR_China = full_LUC(OSCAR_landC)

## filepath
file_pathin = '/home/heyue/OSCAR_China/'

if False:

    ## loops
    For = xr.open_dataset(file_pathin + 'results/n1_test_grid_full_For.nc', chunks={'config': 5})

    for n_run in range(1,101):

        print(n_run)

        Par = xr.open_dataset(file_pathin + f'results/n{n_run}_test_grid_full_Par.nc', chunks={'config': 5})
        Out = xr.open_dataset(file_pathin + f'results/n{n_run}_test_grid_full_Out.nc', chunks={'config': 5})

        Get = lambda var: OSCAR_China[var](Out, Par, For, recursive=True, time_axis='year')

        # cVeg
        cVeg = ((Get('cveg_0') + Get('D_cveg')) * (Par.Aland_0 + Out.D_Aland)).sel(bio_land='Forest').sum('reg_land') + \
            Get('D_Cveg_bk').sum('bio_from', min_count=1).rename({'bio_to':'bio_land'}).sel(bio_land='Forest').sum('reg_land')

        Data = xr.Dataset()
        Data['cVeg'] = cVeg
        Data.to_netcdf(file_pathin + f'/results/postprocess/n{n_run}_cVeg_forConstrain.nc')

    for Var in ['cVeg']:

        print(Var)

        datanews = []

        for n_run in range(1,101):

            file_path = file_pathin + f'results/postprocess/n{n_run}_cVeg_forConstrain.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var]
                datanews.append(datanew)
                del datanew

        datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000))
        datanews.to_netcdf(file_pathin + f'/results/postprocess/{Var}_forConstrain.nc')

    for Var in ['FNET_sum']:

        print(Var)

        datanews = []

        for n_run in range(1,101):

            file_path = file_pathin + f'results/postprocess/n{n_run}_test_grid_full_Out_postprocess.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var]
                datanews.append(datanew)
                del datanew

        datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000))
        datanews.to_netcdf(file_pathin + f'/results/postprocess/{Var}_forConstrain.nc')


    ##  w0：based on RECAAP China FNET, reference values: -0.3006 ± 0.03721
    Data = xr.open_dataset(file_pathin + 'results/postprocess/FNET_sum_forConstrain.nc' )
    period_con = (2000, 2018)
    Fnet_con_av = -0.3006
    Fnet_con_std = 0.03721
    Fnet_con_unc = Data.FNET_sum.sel(year=slice(*period_con)).mean('year')
    w0 = 1 / Fnet_con_std / np.sqrt(2*np.pi) * np.exp(-0.5 * (Fnet_con_unc - Fnet_con_av)**2 / Fnet_con_std**2)
    normalized_w0 = w0 / w0.sum('config')

    ##  w1：based on NFI 9th China cVeg Forest, reference values: 8.98
    Data = xr.open_dataset(file_pathin + 'results/postprocess/cVeg_forConstrain.nc' )
    period_con = (2014, 2018)
    cVeg_con_av = 8.98 # NFI 9th
    cVeg_con_std = 8.98 * 0.20 # 20% uncer
    cVeg_con_unc = Data.cVeg.sel(year=slice(*period_con)).mean('year')
    w1 =  1 / cVeg_con_std / np.sqrt(2 * np.pi) * np.exp(-0.5 * (cVeg_con_unc - cVeg_con_av) ** 2 / cVeg_con_std ** 2)
    normalized_w1 = w1 / w1.sum('config')

    ## w2：combined w1 + w2
    normalized_w2 = (normalized_w0 * normalized_w1)/(normalized_w0 * normalized_w1).sum('config')
    
    ## save
    normalized_w0.to_netcdf(file_pathin + 'results/postprocess/w0.nc') 
    normalized_w1.to_netcdf(file_pathin + 'results/postprocess/w1.nc') 
    normalized_w2.to_netcdf(file_pathin + 'results/postprocess/w2.nc') 



###############################################################################
# constrain 1900-2018 
###############################################################################

## imports 
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


## filepath
file_pathin = '/home/heyue/OSCAR_China/'


## constrained mean and std
def f_Wmean_Wstd_xr(data, weights):

    weights = weights.fillna(0)

    weighted_data = data.weighted(weights)

    mean = weighted_data.mean(dim='config')

    weighted_variance = ((data - mean) ** 2).weighted(weights).mean(dim='config')

    std = np.sqrt(weighted_variance)

    return mean, std

## loop
period = (1900,2018)
def process_allVar(Var, weights, period, cutname='1900_2018'):

    print(Var, period, sep=' | ')

    datanews = [] 

    for n_run in range(1,101):

        file_path = file_pathin + f'results/postprocess/n{n_run}_test_grid_full_Out_postprocess.nc'

        with xr.open_dataset(file_path)as ds:
            datanew = ds[Var].sel(year=slice(*period))
            datanews.append(datanew)
            del datanew
    
    datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000))

    mean, std = f_Wmean_Wstd_xr(datanews, weights)

    del datanews

    new_var_mean = Var + "_mean"
    new_var_std = Var + "_std"

    Data = xr.Dataset()
    Data[new_var_mean] = mean
    Data[new_var_std] = std

    filename = file_pathin + f'results/postprocess/{Var}_test_grid_full_Out_constrain_{cutname}.nc'
    Data.to_netcdf(filename)

    del Data

    return 


weights = xr.open_dataarray(file_pathin + 'results/postprocess/w2.nc')


## all Vars
Vars = ['ELUC_sum','ELUC_pattern','ELUC_Chwp_sum','ELUC_Csoil_sum','ELUC_Cveg_sum',
                    'ELUC_b2b_sum','ELUC_b2b_pattern',
                    'LASC_sum','LASC_Cveg_sum', 'LASC_Csoil_sum',
                    'LSNK_sum','LSNK_Cveg_sum', 'LSNK_Csoil_sum',
                    'LSNKman_sum','LSNKman_pattern',
                    'FNETman_sum','FNETman_pattern',
                    'FNET_sum']

for Var in Vars:
    process_allVar(Var, weights, (1900, 1960), cutname = '1900_1960')
    process_allVar(Var, weights, (1961, 2018), cutname = '1961_2018')
    
print('all var done')


FinalData = xr.Dataset()
for Var in Vars:
    filename1 = file_pathin + f'results/postprocess/{Var}_test_grid_full_Out_constrain_1900_1960.nc'
    filename2 = file_pathin + f'results/postprocess/{Var}_test_grid_full_Out_constrain_1961_2018.nc'

    Data1 = xr.open_dataset(filename1)
    Data2 = xr.open_dataset(filename2)

    Data = xr.concat([Data1, Data2], dim='year')

    FinalData = xr.merge([FinalData, Data])

    del Data

filename =  file_pathin + f'results/postprocess/alldata_test_grid_full_Out_constrain.nc'
FinalData.to_netcdf(filename)


