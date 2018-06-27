import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn.model_selection import train_test_split
from keras.models import load_model
warnings.filterwarnings('ignore')



c = pd.read_csv('extract_ld.csv')

aq = c.iloc[:,2:]


# -------------for 3 attributes
#def get48(aq):
#    
#    a1=[6*i for i in range(35)]
#    a2=[6*i+1 for i in range(35)]   
#    a3=[6*i+4 for i in range(35)]   
#    a4=a1+a2+a3                     
#    a4.sort()
#    g48=aq.iloc[:,a4]
#    g48.columns=[i for i in range(105)]    
#
#    c=[]
#    for i in range(len(aq)-47):
#        te=np.reshape(g48[i:i+48].values,(-1))
#        c.append(te)
#    return np.array(c)

# -------------for 2 attributes
def get48(aq):
    
    a1=[3*i for i in range(13)]
    a2=[3*i+1 for i in range(13)]   

    a4=a1+a2                 
    a4.sort()
    g48=aq.iloc[:,a4]
    g48.columns=[i for i in range(26)]    

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

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()
    
    
    
# --------------- prepare data -----------------    
g48 = get48(aq)

spc = getspc(aq)

tem = gettem(aq)

# calibrated  data : 7972 
g48 = g48[10:]

tem = tem[:6090-47]

spc = spc[9:6100-48]

hold = 400


#tem, _, spc, _, g48, _ =train_test_split(tem, spc,  g48, random_state=42,test_size=0 )


y1 = g48[-hold:]
a1 = tem[-hold:]
a2 = spc[-hold:]


#g48 =np.log( g48)
#tem = np.log(tem)
#spc = np.log(spc)
#me = np.log(me)


g48 = g48[:-hold]
tem = tem[:-hold]
spc = spc[:-hold]



x1_train, x1_test ,\
x2_train,x2_test,  \
y_train, y_test =train_test_split(tem,spc,g48,test_size=0.33, random_state=42)
#
#x1_train =np.log1p( x1_train)
#x2_train = np.log1p(x2_train)
#y_train = np.log1p(y_train)


from model3 import new_model_ld



def smape(actual, predicted):
    a = np.abs(np.array(actual) - np.array(predicted))
    b = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(a, b, out=np.zeros_like(a), where=b!=0, casting='unsafe'))


def train_bj_model(y_train , y_test,x1_train , x1_test, x2_train , x2_test  ,model):
    
    model.compile(optimizer='adam', loss='mae',metrics=['mae','mse'])
    
    lose=model.fit({'tem_input': x1_train, 'spa_input': x2_train, }, [y_train],
                    epochs=60,
                    validation_split=0.2,
                    batch_size=128,
                    shuffle=True,
                    verbose=1)  
    
    score= model.evaluate( x={'tem_input': x1_test, 'spa_input': x2_test}, 
                            y=y_test, 
                            batch_size=30, 
                            verbose=1, 
                            sample_weight=None, 
                            steps=None)
    
    print (score)
    
    pre=(model.predict({'tem_input': x1_test, 'spa_input': x2_test}))
    a=smape(y_test,pre)
    print (smape(y_test,pre))

    
    
    
    np.save('result',a)
    
    return a,score,lose


model= new_model_ld()  

# -----------------------------------train----------------------------

a,b,lose= train_bj_model(y_train , y_test, x1_train , x1_test, x2_train , x2_test ,model)


show_train_history(lose,'loss','val_loss')

c1 = model.predict({'tem_input': a1, 'spa_input': a2 })
c2 = smape(y1,c1)
c3=[]
for i in range(400):
    c3.append(smape(y1[i],c1[i]))





# ------------------------------------save-----------------------------
#s='0524//1st_ld'+'.h5'
#model.save(s)
#0.39



def evalu(foo):

    model=load_model(foo+'.h5')
    pre=(model.predict({'tem_input': x1_test, 'spa_input': x2_test }))
    a=smape(y_test,pre)
    
    print (smape(y_test,pre))
    return a

#score0=evalu('0510//nmeo_1')
#score1=evalu('0510//normal_leak_nomeo_mae_1')
#score2=evalu('0510//leak_nomeo_1')
#score3=evalu('0510//leak_nomeo_2')
#score4=evalu('0510//leak_nomeo_mae_1')
    
# -----------------------pred---------------------------------------

"""


def translate(x):
    a=np.reshape(x,(48,-1))
    a1=a[:,::2]
    a2=a[:,1::2]
    a1=a1.T
    a2=a2.T
    a1=np.reshape(a1,(-1,1))
    a2=np.reshape(a2,(-1,1))
    b=np.hstack([a1,a2])
    return b
    
    
def pred_bj(Tdata):
    spa_tem=load_model('0524//'+'1st_ld'+'.h5')
    
    temp=Tdata.drop(['id', 'station_id', 'time','CO_Concentration', 'O3_Concentration','SO2_Concentration'],axis=1)
    
    temp=np.reshape(temp.values,(1,-1))   # temperal 

    spac=Tdata[130-13:]

    spac=spac.drop(['id', 'station_id', 'time','CO_Concentration', 'O3_Concentration','SO2_Concentration'],axis=1)
    
    spac=np.reshape(spac.values,(1,-1))   # spacial 
    
    pre=spa_tem.predict({'tem_input': temp, 'spa_input': spac })
    
    pre=np.reshape(pre,(-1,2))
    
    pre=translate(pre)
    
    return pre



#[ 'temperature', 'wind_direction', 'wind_speed','weather_Dust', 'weather_Fog',
# 'weather_Haze', 'weather_Rain','weather_Rain with Hail', 
#'weather_Rain/Snow with Hail', 'weather_Sand','weather_Sleet', 'weather_Snow', 'weather_Sunny/clear']


Tdata = pd.read_csv('airquality_ld_2018-05-30-12_2018-06-01-23.csv', delim_whitespace=False) #should be (190,9)




Tdata.drop(Tdata[Tdata['station_id']=='CT3'].index,inplace=True)
Tdata.drop(Tdata[Tdata['station_id']=='CT2'].index,inplace=True)
Tdata.drop(Tdata[Tdata['station_id']=='BX9'].index,inplace=True)
Tdata.drop(Tdata[Tdata['station_id']=='BX1'].index,inplace=True)
Tdata.drop(Tdata[Tdata['station_id']=='RB7'].index,inplace=True)
Tdata.drop(Tdata[Tdata['station_id']=='TD5'].index,inplace=True)


Tdata=Tdata.interpolate()             
Tdata=Tdata.reset_index(drop=True)         # should be (130,9)


Tdata=Tdata.interpolate() 
Tdata=Tdata.fillna(method='bfill')
Tdata=Tdata.fillna(method='ffill')


# predict  
predict=pred_bj(Tdata)
# save prediction 

predict=np.round(predict,decimals=2)
np.savetxt("predictbj.csv", predict, delimiter=",")
"""
# ------------------------------------------------------------------








"""






Tdata = pd.read_csv('airquality_bj_2018-04-01-12_2018-04-02-00.csv', delim_whitespace=False) #should be (350,9)
Tdata=Tdata.interpolate() 
a=Tdata['time'][::35].reset_index()

predict=pred_bj(Tdata)


# save prediction 

predict=np.round(predict,decimals=2)
np.savetxt("predictbj.csv", predict, delimiter=",")



valid=pd.read_csv('airquality_bj_2018-04-02-00_2018-04-04-01.csv')
valid=valid.interpolate() 
valid=valid.fillna(method='backfill')
valid=valid.drop(['id', 'station_id', 'time','O3_Concentration', 'NO2_Concentration','CO_Concentration', 'SO2_Concentration'],axis=1)

p_true=np.array(valid).reshape((1,-1))


result= smape(p_true,predict)

print (result)



"""





















