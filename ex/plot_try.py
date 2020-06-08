#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.getcwd()
os.chdir('csv\\')
# %%
pd.set_option('display.max_columns',None)
df=pd.read_csv('workAblt_V2.csv')
df.drop('zap_town',axis=1,inplace=True)
df.drop('zap_zip',axis=1,inplace=True)


# %%
display(df.head(5))
print("-------------------------------")
df.describe().loc['count':'std',:]

# %%
for i in df.columns:
    display(df[i].value_counts())
# %%
plt.style.use('seaborn-darkgrid')
fig,ax=plt.subplots(1,2,figsize=(12,6),sharey=True)
df[df.sex==1].groupby(['year','low_type_s']).low_type_s.count().unstack().plot.bar(stacked=True,ax=ax[0])
df[df.sex==2].groupby(['year','low_type_s']).low_type_s.count().unstack().plot.bar(stacked=True,ax=ax[1])

ax[0].set(ylabel='count',title='sex=1')
ax[1].set(ylabel='count',title='sex=2')
plt.show()
# %%
plt.figure(figsize=(12,6))
sns.distplot(df[df.sex==1].age ,bins=10,kde=False,label='sex=1')
sns.distplot(df[df.sex==2].age ,bins=10,kde=False,label='sex=2')
plt.legend()
plt.title('sex&age')
plt.show()
# %%
df.groupby(['year']).age.mean().plot()
plt.legend('1')

# %%
def age_sex(row):
    return row['age']-row['sex']

df['new']=df.apply(age_sex, axis=1)
df.loc[0:5,['age','sex','new']]
# %%
