
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.io as scio


folder = '/home/heyue/OSCAR_China/results/postprocess/'

file_pathin  = '/home/heyue/OSCAR_China/'

def xr_reshape(A, dim, newdims, coords):

    ind = pd.MultiIndex.from_product(coords, names=newdims)

    A1 = A.copy()

    A1.coords[dim] = ind

    A1 = A1.unstack(dim)

    i = A.dims.index(dim)
    dims = list(A1.dims)

    for d in newdims[::-1]:
        dims.insert(i, d)

    for d in newdims:
        _ = dims.pop(-1)


    return A1.transpose(*dims)


def f_Wmean_Wstd_xr(data, weights):

    weights = weights.fillna(0)

    weighted_data = data.weighted(weights)

    mean = weighted_data.mean(dim='config')

    weighted_variance = ((data - mean) ** 2).weighted(weights).mean(dim='config')

    std = np.sqrt(weighted_variance)

    return mean, std


weights = xr.open_dataarray(folder + 'w2.nc')

          
################################################## 
## Table1 LUC Strategy
################################################## 

Var = 'ELUC_sum'
SSP = 'SSP1-2.6'

results = {}
for Scen in ['Continued','Accelerated','Pledged']:
    for Strategy in ['Linear','StartFromHigh','StartFromLow']:

        datanews = [] 
        for i in range(1,101):
            n = f'n{i}'
            file_path = folder + n + f'_test_grid_full_Out_{Scen}_{Strategy}_{SSP}_postprocess.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var].load() 
                datanews.append(datanew)
                del datanew

        datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000)).sel(year=slice(2019,2100)).sum('year')

        mean, std = f_Wmean_Wstd_xr(datanews, weights)

        results[(Scen, Strategy)] = f"{mean:.1f}±{std:.1f}"

df1 = pd.DataFrame(index=['Linear'], columns=['Continued','Accelerated','Pledged'])

for Scen in ['Continued','Accelerated','Pledged']:
    for Strategy in ['Linear','StartFromHigh','StartFromLow']:
        df1.at[Strategy, Scen] = results[(Scen, Strategy)]

print(df1)

################################################## 
## Table1 LUC activity
################################################## 

Var = 'ELUC_b2b_sum'
SSP = 'SSP1-2.6'

results = {}
for Scen in ['Continued','Accelerated','Pledged']:
    for Strategy in ['Linear']:
        
        datanews = [] 
        for i in range(1,101):
            n = f'n{i}'
            file_path = folder + n + f'_test_grid_full_Out_{Scen}_{Strategy}_{SSP}_postprocess.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var].load() 
                datanews.append(datanew)
                del datanew

        datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000)).sel(year=slice(2019,2100)).sum('year')

        mean, std = f_Wmean_Wstd_xr(datanews, weights)

        for id, LUCactivity in enumerate(['deforestation', 'other', 'forestation']):

            results[(Scen, LUCactivity)] = f"{mean[id]:.1f}±{std[id]:.1f}"

df2 = pd.DataFrame(index=['deforestation', 'other', 'forestation'],columns=['Continued','Accelerated','Pledged'])

for Scen in ['Continued','Accelerated','Pledged']:
    for LUCactivity in ['deforestation', 'other', 'forestation']:
        df2.at[LUCactivity, Scen] = results[(Scen, LUCactivity)]

print(df2)

############################################################### 
## Table1 LULUCF direct/indirect/total and Contributions
############################################################### 

Strategy = 'Linear'
SSP = 'SSP1-2.6'

results = {} 
for SSP in ['SSP1-2.6']:
    for Scen in ['Continued','Accelerated','Pledged']:

        ## managed forest area
        NonIntactForestArea = xr.open_dataset(file_pathin + f'/input_data/NonIntactForestArea_1900_2100.nc')
        NonIntactForestArea_stacked = NonIntactForestArea[Scen].sel(year=slice(2019,2100))


        ################# indirect flux
        Var = 'D_nbp'

        datanews = [] 
        for i in range(1,101):
            n = f'n{i}'
            file_path = folder + n + f'_test_grid_full_Out_{Scen}_{Strategy}_{SSP}_postprocess.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var].load() 
                datanews.append(datanew)
                del datanew

        datanews = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000))

        # indirect 1000 config => 2019-2100 sum
        indirect = (datanews.sel(bio_land='Forest') * NonIntactForestArea_stacked).sum(dim=['reg_land'])
        indirect_mean, indirect_std = f_Wmean_Wstd_xr(indirect.sum(dim=['year']), weights)

        # indirect 1000 config => 2055-2065 mean
        indirect_2060 = (datanews.sel(bio_land='Forest') * NonIntactForestArea_stacked).sum(dim=['reg_land']).sel(year=slice(2055,2065)).mean('year')
        indirect_2060_mean, indirect_2060_std = f_Wmean_Wstd_xr(indirect_2060 , weights)


        ################# direct flux
        Var = 'ELUC_sum'

        datanews = [] 
        for i in range(1,101):
            n = f'n{i}'
            file_path = folder + n + f'_test_grid_full_Out_{Scen}_{Strategy}_{SSP}_postprocess.nc'

            with xr.open_dataset(file_path)as ds:
                datanew = ds[Var].load() 
                datanews.append(datanew)
                del datanew

        direct = xr.concat(datanews, dim='config').assign_coords(config=np.arange(1000))

        # direct 1000 config => 2019-2100 sum
        direct_mean, direct_std = f_Wmean_Wstd_xr(direct.sum(dim=['year']), weights)

        # direct 1000 config => 2055-2065 mean
        direct_2060 = direct.sel(year=slice(2055,2065)).mean('year')
        direct_2060_mean, direct_2060_std = f_Wmean_Wstd_xr(direct_2060, weights)


        ################# total flux
        # total 1000 config  => 2019-2100 sum
        total = indirect + direct
        total_mean, total_std = f_Wmean_Wstd_xr(total.sum(dim=['year']), weights)

        # total 1000 config  => 2055-2065 mean
        total_2060 = indirect_2060 + direct_2060
        total_2060_mean, total_2060_std = f_Wmean_Wstd_xr(total_2060, weights)

        ################# Contribution
        Contribution_direct_mean, Contribution_direct_std = f_Wmean_Wstd_xr(direct.sum(dim=['year'])/total.sum(dim=['year']), weights)
        Contribution_indirect_mean, Contribution_indirect_std = f_Wmean_Wstd_xr(indirect.sum(dim=['year'])/total.sum(dim=['year']), weights)

        ################# save
        Data = xr.Dataset()

        # 2019-2100 sum
        Data['indirect_mean'] = indirect_mean
        Data['indirect_std'] = indirect_std
        Data['direct_mean'] = direct_mean
        Data['direct_std'] = direct_std
        Data['total_mean'] = total_mean
        Data['total_std'] = total_std
        
        # 2055-2065 mean
        Data['indirect_2060_mean'] = indirect_2060_mean
        Data['indirect_2060_std'] = indirect_2060_std
        Data['direct_2060_mean'] = direct_2060_mean
        Data['direct_2060_std'] = direct_2060_std
        Data['total_2060_mean'] = total_2060_mean
        Data['total_2060_std'] = total_2060_std
        
        # contribution of indirect/direct to total,2019-2100 sum 
        Data['Contribution_indirect_mean'] = Contribution_indirect_mean *100
        Data['Contribution_indirect_std'] = Contribution_indirect_std *100
        Data['Contribution_direct_mean'] = Contribution_direct_mean *100
        Data['Contribution_direct_std'] = Contribution_direct_std *100
        
        results[('direct', Scen)] = f"{Data['direct_mean']:.1f}±{Data['direct_std']:.1f}"
        results[('indirect', Scen)] = f"{Data['indirect_mean']:.1f}±{Data['indirect_std']:.1f}"
        results[('total', Scen)] = f"{Data['total_mean']:.1f}±{Data['total_std']:.1f}"

        results[('direct_2060', Scen)] = f"{Data['direct_2060_mean']:.2f}±{Data['direct_2060_std']:.2f}"
        results[('indirect_2060', Scen)] = f"{Data['indirect_2060_mean']:.2f}±{Data['indirect_2060_std']:.2f}"
        results[('total_2060', Scen)] = f"{Data['total_2060_mean']:.2f}±{Data['total_2060_std']:.2f}"

        results[('Contribution_direct', Scen)] = f"{Data['Contribution_direct_mean']:.1f}±{Data['Contribution_direct_std']:.1f}"
        results[('Contribution_indirect', Scen)] = f"{Data['Contribution_indirect_mean']:.1f}±{Data['Contribution_indirect_std']:.1f}"
        
        # offset 2060 hard-to-abate emission
        results[('direct_2060_offset', Scen)] = f"{(Data['direct_2060_mean']*100/0.82):.1f}±{(Data['direct_2060_std']*100/0.82):.1f}"
        results[('indirect_2060_offset', Scen)] = f"{(Data['indirect_2060_mean']*100/0.82):.1f}±{(Data['indirect_2060_std']*100/0.82):.1f}"
        results[('total_2060_offset', Scen)] = f"{(Data['total_2060_mean']*100/0.82):.1f}±{(Data['total_2060_std']*100/0.82):.1f}"


List_LULUCF = ['direct', 'indirect', 'total', 'direct_2060', 'indirect_2060', 'total_2060', 'direct_2060_offset', 'indirect_2060_offset', 'total_2060_offset', 'Contribution_direct', 'Contribution_indirect']
df3 = pd.DataFrame(index=List_LULUCF,columns=['Continued','Accelerated','Pledged'], )

for Scen in ['Continued','Accelerated','Pledged']:
    for LUC in List_LULUCF:
        df3.at[LUC, Scen] = results[(LUC, Scen)]

print(df3)

############################################################### 
## Table1 
############################################################### 

df_concatenated = pd.concat([df1, df2, df3], axis=0) 

print(df_concatenated)

df_concatenated.to_csv(file_pathin + 'Table 1.csv')

