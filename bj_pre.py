
from check_meo import get_data,get_station_aq,get_48,get_temperal,smape
import pandas as pd
from model import build_auto_m,n_model

bj_meo='Beijing_historical_meo_grid.csv'
bj_aq='beijing_17_18_aq.csv'
station1='station.csv'

# load meo , aq for training
bj=pd.read_csv(bj_aq)
bjm=pd.read_csv(bj_meo)
station=pd.read_csv(station1 , header=None)
bjma=pd.read_csv('clean_meo.csv')

meo, aq ,meoo = get_data(bjm, bj, station ,bjma)


# do interpolate
aq=aq.interpolate()

# set index1- stations :
station = pd.read_csv(station,header=None)
station_aq=get_station_aq(station)




def train_m(aq,station):
    a,b=[],[]
    for i in range(len(station)):
        
        test_1st = station[i]            # test first station
        print ('step:', i, 'running ', test_1st , '...')
        
        p_48=get_48(aq,test_1st)
        tem=get_temperal(aq,test_1st)
        
        ydata=p_48[20:]                 # labe
        x1data=aq[19:9482-49].values    # spacial
        x2data=tem[0:9482-68]           # temperal
        x3data=meoo[19:9482-49].values  # meoo
        
        spac_train, spac_test= x1data[:7000],x1data[7000:]
        temp_train, temp_test= x2data[:7000],x2data[7000:]
        meoo_train, meoo_test= x3data[:7000],x3data[7000:]
        labe_train, labe_test = ydata[:7000],ydata[7000:]
        
        model = n_model()
        cost= model.fit({'spa_input': spac_train, 'tem_input': temp_train,'meoo_input': meoo_train}, [labe_train],
                        epochs=100,
                        validation_split=0.2,
                        batch_size=200,
                        shuffle=True,
                        verbose=1)
        
        score= model.evaluate( x={'spa_input': spac_test, 'tem_input': temp_test,'meoo_input': meoo_test}, 
                                y=labe_test, 
                                batch_size=30, 
                                verbose=1, 
                                sample_weight=None, 
                                steps=None)
        print (score)
        b.extend([score])
        
        
        pre=(model.predict({'spa_input': spac_test, 'tem_input': temp_test,'meoo_input': meoo_test}))
        a.extend([smape(labe_test,pre)])
        print (smape(labe_test,pre))
        s='a0508//'+test_1st+'.h5'
        model.save(s)
    return a,b


#   train 
a , b= train_m (aq,station_aq)  
    
# 用矩陣可定義 loss 



#auto, enco = build_auto_m()

#auto.fit( x= meo, y=meo , batch_size=50, epochs=10000, verbose=1, )


#鵽a=pd.read_csv('temp_data.csv')


