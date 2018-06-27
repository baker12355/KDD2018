import pandas as pd
import numpy as np


def get_station_aq(station):
    station_aq=list()
    for i in station[0]:
        station_aq.append(str(i)+'_aq')
    station_aq[3]='wanshouxigong_aq'
    station_aq[4]='aotizhongxin_aq'
    station_aq[5]='nongzhanguan_aq'
    station_aq[9]='fengtaihuayuan_aq'
    station_aq[25]='miyunshuiku_aq'
    station_aq[31]='yongdingmennei_aq'
    station_aq[32]='xizhimenbei_aq'
    station_aq=station_aq[0:35]
    return station_aq

def get_ordered_table(bja,station):
    ordered = pd.DataFrame()
    for i in station:
        a=bja[bja['stationId']==i]
        a=a.drop(['stationId','utc_time','NO2','CO','SO2'],axis=1)
        a.reset_index(drop=True,inplace=True)
        ordered=pd.concat([ordered,a],axis=1,ignore_index=True)
    return ordered


def get_48(aq,station):
    a=aq[station].values
    b=np.array([])
    for j in range(0,9482-48):
        b=np.append(b, a[j:j+48])
    c=np.reshape(b,(-1,48*3))
    return c

def get_temperal(aq,station):
    a=aq[station].values
    b=np.array([])
    for j in range(20,9482):
        b=np.append(b,a[j-20:j])
    c=np.reshape(b,(-1,60))
    return c


def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))

# col= ['stationName', 'longitude', 'latitude', 'utc_time', 'temperature',
#       'pressure', 'humidity', 'wind_direction', 'wind_speed/kph']

#651 stations, each station 10806 data from 2017-1-1 to 2018-3-27 05:00:
    
"""
bj_meo='Beijing_historical_meo_grid.csv'
bj_aq='beijing_17_18_aq.csv'
station_1='station.csv'
"""
def get_data(bjm,bj_aq,station,bjma):
    
    

    
    utc=bjm['utc_time'][::651].reset_index(drop=True)
    station_aq=get_station_aq(station)
    
    bjm.drop(['stationName', 'longitude','utc_time', 'latitude','pressure', 'humidity',],axis=1,inplace=True)
    
    bj=np.reshape(bjm.values,(-1,651*3))
    bj=pd.DataFrame(bj,)
    bj=pd.concat([bj,utc],axis=1)
    
    # 2017-01-01 14:00:00 2018-01-31 15:00:00
    
    bj_airquality = pd.read_csv(bj_aq, delim_whitespace=False)
    bj_airquality = bj_airquality.drop_duplicates()
    
    
    ordered=get_ordered_table(bj_airquality,station_aq)
    b=bj_airquality['utc_time'][:8701].reset_index(drop=True)
    ordered=pd.concat([ordered,b],axis=1)
    ordered=ordered.interpolate()
    
    # concated
    result = pd.merge(bj, ordered , how='left', on='utc_time')
    
    
    # read  meoo

    
    
    bjma= unify_utc(bjma)
    bjma = bjma.drop_duplicates()
    # getorderd bjma
    a=bjma['station_id']
    a=bjma['station_id'][::8781]
    sii=a.values
    a=bjma[bjma['station_id']=='shunyi_meo']
    bjma= pd.get_dummies(data=bjma,columns=['weather'])
    
    #bjma= bjma.drop([ 'longitude', 'latitude', 'temperature','pressure', 'humidity', 'wind_direction', 'wind_speed'],axis=1)
    bjma= bjma.drop([ 'longitude', 'latitude' ,'pressure', 'humidity', 'temperature', 'wind_direction', 'wind_speed','weather_Dust', 'weather_Fog', 'weather_Haze', 'weather_Snow', 'weather_Sunny/clear' ],axis=1)
    
    
    
    te=bjma[bjma['station_id']==sii[0]]
    te=te.drop(['station_id'], axis=1)
    for i in sii:
        if i!=sii[0]:
            t=bjma[bjma['station_id']==i]
            t=t.drop(['station_id'] ,axis=1)
            t=t.reset_index(drop=True,)
            te=pd.merge(te,t,how='left',on='utc_time')
    
    # merge  moe + aq + meoo
    utc=bjma['utc_time'][::18].reset_index(drop=True)
    result = pd.merge(result, te , how='left', on='utc_time')
    
    result=result.fillna(0)
    
    
    # 14 9496 we have integrated meo and aq (1953 + 1 + 105 )= 2059 ,* 9482  +180 
    
    result=result.iloc[14:9496]
    
    meo = result.iloc[:,:1953]
    aq = result.iloc[:,1954:1954+105]
    meoo =result.iloc[:,1954+105:]
    
    # set aq label
    c=[]
    for i in station_aq:
        for j in range(3):
            c.append(i)
    aq.columns=c
    
    # reset index
    aq.reset_index(inplace=True,drop=True)
    meo.reset_index(inplace=True,drop=True)
    return meo,aq,meoo




# utc time does not in a same type

def unify_utc(bjm):
    bjm['utc_time']=bjm['utc_time']+':00'
    bjmt=np.array(bjm['utc_time'].values,dtype=str)
    bjmt2=[]
    for i,j in enumerate(bjmt):
        tem=j.split(' ')
        a=tem[0].split('/')
        bjmt2.append(tem[1])
        if len(a[1])<2:
            a[1]='0'+a[1]
        if len(a[2])<2:
            a[2]='0'+a[2]
        tem='/'.join(a)
        bjmt[i]=tem
        bjmt[i]+=' '+bjmt2[i]
        bjmt[i]=bjmt[i].replace('/','-')
    
    bjm['utc_time']=bjmt
    return bjm







"""
#861 stations, each station 10806 data from 2017-1-1 to 2018-3-27 05:00:00
lda=pd.read_csv('London_historical_meo_grid.csv')
utc=lda['utc_time'][::861].reset_index(drop=True)
lda.drop(['stationName', 'longitude','utc_time', 'latitude','pressure', 'humidity',],axis=1,inplace=True)


ld_a = pd.read_csv('London_historical_aqi_forecast_stations_20180331.csv', delim_whitespace=False)
stationld = pd.read_csv('stationld.csv',header=None)[0].values
stationld = np.array(stationld,dtype=str)
ld_a=ld_a.interpolate()

"""