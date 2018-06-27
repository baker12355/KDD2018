# This first set of packages include Pandas, for data manipulation, numpy for mathematical computation and matplotlib & seaborn, for visualisation.
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')

# Statistical packages used for transformations
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats.stats import pearsonr
print('Statistical packages imported!')

# Metrics used for measuring the accuracy and performance of the models
#from sklearn import metrics
#from sklearn.metrics import mean_squared_error
print('Metrics packages imported!')

# Algorithms used for modeling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
print('Algorithm packages imported!')

# Pipeline and scaling preprocessing will be used for models that are sensitive
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print('Pipeline and preprocessing packages imported!')

# Model selection packages used for sampling dataset and optimising parameters
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
print('Model selection packages imported!')

# Set visualisation colours
mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are ready! :)')

# To ignore annoying warning
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print('Deprecation warning will be ignored!')




# ---------------------------------------------------------------------------

aq = pd.read_csv('beijing_17_18_aq.csv')
meo = pd.read_csv('clean_meo.csv')

station_aq=['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq'
,'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq','yungang_aq'
,'gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq','shunyi_aq','pingchang_aq'
,'mentougou_aq','pinggu_aq','huairou_aq','miyun_aq','yanqin_aq','dingling_aq','badaling_aq'
,'miyunshuiku_aq','donggaocun_aq','yongledian_aq','yufa_aq','liulihe_aq',
'qianmen_aq','yongdingmennei_aq','xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']


meo = meo.drop(['longitude', 'latitude', 'pressure',], axis=1 )

#------------------------Drop na duplicated

b=aq.loc[:,['utc_time','stationId']].duplicated()

aq = aq.drop(b[b==True].index)


# -------------------------ouliers----------------------------------------
plt.subplots(figsize=(15,5))
plt.subplot(1, 2, 1)
g = sns.regplot(x=aq['PM10'], y=aq['PM2.5'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
aq = aq.drop(aq[(aq['PM10']>1000)].index)
g = sns.regplot(x=aq['PM10'], y=aq['PM2.5'], fit_reg=False).set_title("After")

# -------------------------ouliers----------------------------------------
plt.subplots(figsize=(15,5))
plt.subplot(1, 2, 1)
g = sns.regplot(x=aq['NO2'], y=aq['PM2.5'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
aq = aq.drop(aq[aq['NO2']>290].index)
aq = aq.drop(aq[aq['PM2.5']>800].index)
g = sns.regplot(x=aq['NO2'], y=aq['PM2.5'], fit_reg=False).set_title("After")

# -------------------------ouliers----------------------------------------
plt.subplots(figsize=(15,5))
plt.subplot(1, 2, 1)
g = sns.regplot(x=aq['SO2'], y=aq['PM2.5'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
aq = aq.drop(aq[aq['SO2']>200].index)
g = sns.regplot(x=aq['SO2'], y=aq['PM2.5'], fit_reg=False).set_title("After")

# -------------------------ouliers----------------------------------------
plt.subplots(figsize=(15,5))
plt.subplot(1, 2, 1)
g = sns.regplot(x=aq['CO'], y=aq['PM2.5'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                   
a= set(aq[aq['CO']>11].index.values )
b= set(aq[aq['PM2.5']<300].index.values )
c= a or b
aq = aq.drop(c)
g = sns.regplot(x=aq['CO'], y=aq['PM2.5'], fit_reg=False).set_title("After")

# -------------------------ouliers----------------------------------------
plt.subplots(figsize=(15,5))
plt.subplot(1, 2, 1)
g = sns.regplot(x=aq['O3'], y=aq['PM2.5'], fit_reg=False).set_title("Before")

# Delete outliers
plt.subplot(1, 2, 2)                                                                                
aq = aq.drop(aq[aq['O3']>450].index)
g = sns.regplot(x=aq['O3'], y=aq['PM2.5'], fit_reg=False).set_title("After")


# ------------------------merge by Date-----------------------------------


rng = pd.date_range(start='2017-01-01 14:00:00',end='2018-01-31 15:00:00', freq='H')
a = pd.DataFrame(data=rng,columns=['utc_time'])

# to utc_time

aq['utc_time'] = pd.to_datetime(aq['utc_time'])
# merge
for i in station_aq:
    t = aq[aq['stationId']==i].drop(['stationId'],axis=1)
    #t = aq[aq['stationId']==i]
    a=pd.merge(a ,t, how='left', on= 'utc_time')




a=a.interpolate()

a=a.fillna( method= 'bfill')

a=a[730:-16]










# -------------------------meo processing --------------------------------

#['station_id', 'utc_time', 'temperature', 'wind_direction', 'wind_speed',
#       'weather_Dust', 'weather_Fog', 'weather_Haze', 'weather_Rain',
#       'weather_Rain with Hail', 'weather_Rain/Snow with Hail', 'weather_Sand',
#       'weather_Sleet', 'weather_Snow', 'weather_Sunny/clear']

meo['utc_time'] = pd.to_datetime(meo['utc_time'])

meo = meo.interpolate()

z = np.array(meo['station_id'].unique(),dtype=str)

# dummies
meo = pd.get_dummies(meo, columns = ["weather"], prefix="weather")

b = pd.DataFrame(data=rng,columns=['utc_time'])

for i in z :
    t = meo[meo['station_id']==i].drop(['station_id'],axis=1)
    a=pd.merge(a ,t, how='left', on= 'utc_time')



c=a


c = a.dropna(thresh=2)



# hold 8400 
c = c.dropna(thresh=460)

c = a.dropna()


c = c.interpolate()
c = c.fillna( method= 'bfill')
c = c.fillna( method= 'ffill')



c = c.drop(['utc_time'], axis=1 )

c.to_csv('no_extract.csv' , index=None)












# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------










































































































































































