import pandas as pd
import numpy as np




bj_meteorology = pd.read_csv('beijing_17_18_meo.csv', delim_whitespace=False)


data=bj_meteorology

data=data.interpolate()       #interpolate to missing data

for i in range(data.index.max()):
    if data.loc[i,'temperature']>80:
        data.loc[i,'temperature']=np.nan
        
    if data.loc[i,'humidity']>200:
        data.loc[i,'humidity']=np.nan    
        
    if data.loc[i,'pressure']>1500:
        data.loc[i,'pressure']=np.nan
        
    if data.loc[i,'wind_direction']<0 or data.loc[i,'wind_direction']>360:
        data.loc[i,'wind_direction']=np.nan
        
    if data.loc[i,'wind_speed']>10:
        data.loc[i,'wind_speed']=np.nan
    
    if i % 1000==0:
        print('已清理存在异常值 %s 行数据'%i) 
    
print(data.describe().astype(np.int64).T) 

data=data.interpolate()

data.to_csv('clean_meo.csv',index=None) 









































