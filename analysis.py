import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


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
        a=a.drop(['stationId','utc_time'],axis=1)
        a.reset_index(drop=True,inplace=True)
        ordered=pd.concat([ordered,a],axis=1,ignore_index=True)
    return ordered


"""
# station setup
station = pd.read_csv('station.csv',header=None)
station_aq=get_station_aq(station) 


df = pd.read_csv('beijing_17_18_aq.csv')
df = df.interpolate()


ddf = pd.read_csv('clean_meo.csv')



# unify utc_time
ddf_time=pd.to_datetime(ddf['utc_time'], format='%Y/%m/%d')
ddf=ddf.drop(['longitude', 'latitude', 'pressure', 'humidity','weather',],axis=1)
ddf['utc_time']=ddf_time


#------------------------------------------

# ordered date and append date
ordered = get_ordered_table(df,station_aq)
utctime = df['utc_time'][:8886]
utctime = pd.to_datetime(utctime, format='%Y/%m/%d')

a=utctime.duplicated()
b=a[a==True].index

# figure out a way to make a 8701*35*6 array

ordered = pd.concat([ordered,utctime], axis=1)

ordered = ordered.drop(b)

#----------------------------------------


# meo cleaning --------------------------
meo_sta=ddf['station_id'][::8781]
a=pd.DataFrame(ddf[ddf['station_id']=='shunyi_meo']['utc_time'])

for i in meo_sta:
    
    b = ddf[ddf['station_id']==i].drop(['station_id'],axis=1)
    b = pd.get_dummies(b)
    a=pd.merge(a,b,how='inner',on='utc_time')


#---------------------------------------

# time synchronization
rng = pd.DataFrame(pd.date_range(start='2017-01-01 14:00:00',end='2018-01-31 15:00:00', freq='H'),columns=['utc_time'])


b = pd.merge(rng ,ordered ,how='inner',on=['utc_time'])


#---------------------------------------  c is 8775 * ( 1 + 210 + 54 )
c = pd.merge(b , a , how='inner', on=['utc_time'])
c = c.reset_index(drop=True)
c.to_csv('data.csv',index=None)

#-----------------------seperat aq meo
"""

c = pd.read_csv('data.csv')

aq = c.iloc[:,1:211]
meo = c.iloc[:,211:]


def get48(aq):
    a1=[6*i for i in range(35)]
    a2=[6*i+1 for i in range(35)]
    a3=[6*i+4 for i in range(35)]
    a4=a1+a2+a3
    a4.sort()
    g48=aq.iloc[:,a4]
    g48.columns=[i for i in range(105)]    
    c=[]
    for i in range(len(aq)-47):
        te=np.reshape(g48[i:i+48].values,(-1))
        c.append(te)
    return np.array(c)

def getspc(aq):
    return aq.values

def gettem(aq):
    c=[]
    for i in range(10,len(aq)-10):
        te=np.reshape(aq[i-10:i].values,(-1))
        c.append(te)
    return np.array(c)


def getmeo(meo):
    return meo.values


g48 = get48(aq)

spc = getspc(aq)

tem = gettem(aq)

me = getmeo(meo)

# calibrated  data : 7972 
g48 = g48[10:]

tem = tem[:8009-37]

spc = spc[9:8029-48]

me = me [9:8029-48]


y_train , y_test = g48[:6300],g48[6300:]

x1_train , x1_test = tem[:6300],tem[6300:]

x2_train , x2_test = spc[:6300],spc[6300:]

x3_train , x3_test = me[:6300],me[6300:]


#x1_train, x1_test ,\
#x2_train,x2_test,  \
#x3_train , x3_test,\
#y_train, y_test =train_test_split(tem,spc,me,g48,test_size=0.33, random_state=42)
      


from model import new_model

model = new_model()

model.fit({'tem_input': x1_train, 'spa_input': x2_train, 'meo_input': x3_train }, [y_train],
                epochs=100,
                validation_split=0.2,
                batch_size=128,
                shuffle=True,
                verbose=1)

score= model.evaluate( x={'tem_input': x1_test, 'spa_input': x2_test, 'meo_input': x3_test }, 
                        y=y_test, 
                        batch_size=30, 
                        verbose=1, 
                        sample_weight=None, 
                        steps=None)



def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))


def train_bj_model(y_train , y_test,x1_train , x1_test,x2_train , x2_test ,x3_train , x3_test ):
    
    model= new_model()     
    
    model.fit({'tem_input': x1_train, 'spa_input': x2_train, 'meo_input': x3_train }, [y_train],
                    epochs=80,
                    validation_split=0.2,
                    batch_size=128,
                    shuffle=True,
                    verbose=1)  
    
    score= model.evaluate( x={'tem_input': x1_test, 'spa_input': x2_test, 'meo_input': x3_test }, 
                            y=y_test, 
                            batch_size=30, 
                            verbose=1, 
                            sample_weight=None, 
                            steps=None)

    print (score)
    
    pre=(model.predict({'tem_input': x1_test, 'spa_input': x2_test, 'meo_input': x3_test }))
    a=smape(y_test,pre)
    print (smape(y_test,pre))

    s='0509//' '1st'+'.h5'
    
    model.save(s)
    np.save('result',a)
    
    return a,score

a,b= train_bj_model(y_train , y_test,x1_train , x1_test,x2_train , x2_test ,x3_train , x3_test )
























