#%%
# oracle

import os
import pandas as pd
import time
from sqlalchemy import create_engine

os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.UTF8'

engine = create_engine('oracle://socsusr:socsusr1234@192.168.1.140:1521/testdb12c',encoding ='UTF-8',echo=False)

connection = engine.raw_connection() 
sql ='''
select * from A1D_LOW_TYPE_ALL
'''
Stime=time.time()
df=pd.read_sql(sql,engine)
Etime=time.time()
print(Etime-Stime)

#%%
A1D=['app_no','year','nation','age','sex','position','low_type_s','applied','helped','grant_yn'
     ,'veteran','settled','aborigine','foreigner','foreigner_child','single_type','single_parent','single_child',
    'next_geration','single_live','sick_card','spc_wom','tranjob_yn','graduate_yn','crip_level','marry','job_code',
    'except_yn','work_ablt','zap_zip','town_name']

#%%
#POSIT,'zap_zip'ION
df.position = df.position.replace('01','0')
df.position = df.position.replace(['02','03','04','05', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', 
'22', '23', '24', '26', '27', '28', '29', '53', '54', '55', '56', '63', '64', '67', '68', '69', '70', '73', '74', '75', 
'78', '79', '80', '81', '82', '83', '84', '85', '89'],'1')
df.position = df.position.replace(['06', '07', '08', '09', '25', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
'46', '47', '48', '49', '58', '59', '60'],'2')
df.position = df.position.replace(['00', '10', '30', '31', '32', '33', '34', '50', '51', '52', '57', '61', '62', '65', '66', 
'71', '72', '76', '77', '86', '87', '88', '98', '99'],'3')
df.position = df.position.fillna('3')
df.position = df.position.astype('str')

#%%
df.marry.fillna(value='9',inplace=True)
df.marry=df.marry.astype('str')  

#%%
df=df.drop(df[df.except_code=='98'].index)
df=df.drop(df[(df.nojob_desc.str.find('死亡')>=0)|(df.job_desc.str.find('死亡')>=0)].index)
df.drop(df[(df.age<0)|(df.age.isnull())].index,inplace=True)


#%%
df.single_type.fillna(value=0,inplace=True)
df['single_type']=df.single_type.replace('N',0)
df['single_type']=df.single_type.replace('Y',1)

df.crip_level.fillna(value=0,inplace=True)
df['crip_level']=df.crip_level.replace('5',0).astype('int')

#%%
df.loc[df.zap_town=='6600006000','zap_zip']='407'    
df.loc[df.zap_town=='6600009000','zap_zip']='420'
df.loc[df.zap_town=='6600018000','zap_zip']='428'
df.loc[df.zap_town=='6600024000','zap_zip']='432'
df.loc[df.zap_town=='6600029000','zap_zip']='424'
df.loc[df.zap_town=='6600011000','zap_zip']='437'
df.loc[df.zap_town=='6600016000','zap_zip']='429'
df.loc[df.zap_town=='6600021000','zap_zip']='438'
df.loc[df.zap_town=='6600007000','zap_zip']='408'
df.loc[df.zap_town=='6600008000','zap_zip']='406'
df.loc[df.zap_town=='6600025000','zap_zip']='434'
df.loc[df.zap_town=='6600027000','zap_zip']='411'
df.loc[df.zap_town=='6600012000','zap_zip']='436'
df.loc[df.zap_town=='6600001000','zap_zip']='400'
df.loc[df.zap_town=='6600005000','zap_zip']='404'
df.loc[df.zap_town=='6600013000','zap_zip']='433'
df.loc[df.zap_town=='6600026000','zap_zip']='413'
df.loc[df.zap_town=='6600002000','zap_zip']='401'
df.loc[df.zap_town=='6600004000','zap_zip']='403'
df.loc[df.zap_town=='6600019000','zap_zip']='426'
df.loc[df.zap_town=='6600020000','zap_zip']='422'
df.loc[df.zap_town=='6600023000','zap_zip']='414'
df.loc[df.zap_town=='6600003000','zap_zip']='402'
df.loc[df.zap_town=='6600010000','zap_zip']='423'
df.loc[df.zap_town=='6600014000','zap_zip']='435'
df.loc[df.zap_town=='6600017000','zap_zip']='427'
df.loc[df.zap_town=='6600022000','zap_zip']='439'
df.loc[df.zap_town=='6600015000','zap_zip']='421'
df.loc[df.zap_town=='6600028000','zap_zip']='412'

df['town_name']=df.zap_zip.astype('str')
df.loc[df.town_name=='400','town_name']='中區'
df.loc[df.town_name=='401','town_name']='東區'
df.loc[df.town_name=='402','town_name']='南區'
df.loc[df.town_name=='403','town_name']='西區'
df.loc[df.town_name=='404','town_name']='北區'
df.loc[df.town_name=='406','town_name']='北屯區'
df.loc[df.town_name=='407','town_name']='西屯區'
df.loc[df.town_name=='408','town_name']='南屯區'
df.loc[df.town_name=='411','town_name']='太平區'
df.loc[df.town_name=='412','town_name']='大里區'
df.loc[df.town_name=='413','town_name']='霧峰區'
df.loc[df.town_name=='414','town_name']='烏日區'
df.loc[df.town_name=='420','town_name']='豐原區'
df.loc[df.town_name=='421','town_name']='后里區'
df.loc[df.town_name=='422','town_name']='石岡區'
df.loc[df.town_name=='423','town_name']='東勢區'
df.loc[df.town_name=='424','town_name']='和平區'
df.loc[df.town_name=='426','town_name']='新社區'
df.loc[df.town_name=='427','town_name']='潭子區'
df.loc[df.town_name=='428','town_name']='大雅區'
df.loc[df.town_name=='429','town_name']='神岡區'
df.loc[df.town_name=='432','town_name']='大肚區'
df.loc[df.town_name=='433','town_name']='沙鹿區'
df.loc[df.town_name=='434','town_name']='龍井區'
df.loc[df.town_name=='435','town_name']='梧棲區'
df.loc[df.town_name=='436','town_name']='清水區'
df.loc[df.town_name=='437','town_name']='大甲區'
df.loc[df.town_name=='438','town_name']='外埔區'
df.loc[df.town_name=='439','town_name']='大安區'
#%%
df.loc[df.job_code=='00','job_code']=0
df.loc[df.job_code=='95','job_code']=0
df.loc[df.job_code=='96','job_code']=0
df.loc[df.job_code=='97','job_code']=0
df.loc[df.job_code=='98','job_code']=0
df.loc[df.job_code!=0,'job_code']=1
df.job_code=df.job_code.astype('str')
#%%
df.work_ablt.fillna(value='00',inplace=True)
df.work_ablt=df.work_ablt.astype('str')
#%%
df[A1D].to_csv('a1d_test.csv',encoding = 'utf8',index=False)


#%%
# postgresql
import os
import pandas as pd
import time
from sqlalchemy import create_engine

os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.UTF8'

engine = create_engine('postgresql://{username}:{password}@{hostname}:{port}/{service_name}'.format(
    username='taibdusr',
    password='taibdusr1234',
    hostname='192.168.1.140',
    port='5432',
    service_name='bigdata'
),encoding ='UTF-8'
)
connection = engine.raw_connection() 
sql ='''
SELECT * from low_type_home_101_107
UNION 
SELECT * FROM low_type_home_108
'''
Stime=time.time()
df=pd.read_sql(sql,engine)
Etime=time.time()
print(Etime-Stime)

#%%
df.to_csv('a1m_test.csv',encoding = 'utf8',index=False)


#%%
df.info()


#%%



