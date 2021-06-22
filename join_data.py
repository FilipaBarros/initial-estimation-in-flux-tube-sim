import pandas as pd
import os

data = pd.DataFrame(columns=['R[Rsun]','L[Rsun]','lon[Carr]','lat[Carr]','B[G]','A/A0','alpha[deg]','V/Cs','propag_dt[d]','n[cm^-3]','v[km/s]','T [MK]'])

def data_concat(path): 
    global data 
    df = pd.read_csv(path)
    row = df.sample(1).values.flatten()[1:]
    df2 = pd.Series(row, index = data.columns)
    df2 = df2.to_frame().T
    data = pd.merge(data,df2,how='outer')
    #append 64 random rows of each dataframe
    

def files_iteration(directory): 
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            data_concat(os.path.join(directory, filename))
            print(filename)
        else:
            continue



files_iteration(r'../MULTI_VP_profiles/profiles_wso_CR1992')
files_iteration(r'../MULTI_VP_profiles/profiles_wso_CR2056')
files_iteration(r'../MULTI_VP_profiles/profiles_wso_CR2071')
files_iteration(r'../MULTI_VP_profiles/profiles_wso_CR2125')
files_iteration(r'../MULTI_VP_profiles/profiles_wso_CR2210')

data.to_csv(r'merged_new2_data.csv', index = False)