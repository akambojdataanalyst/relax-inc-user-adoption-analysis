import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
u=pd.read_csv(DATA/'takehome_users.csv',encoding='latin1'); e=pd.read_csv(DATA/'takehome_user_engagement.csv',encoding='latin1')
e['day']=pd.to_datetime(e.time_stamp).dt.normalize(); e=e.drop_duplicates(['user_id','day']).sort_values(['user_id','day'])
# An adoption event is any 3 distinct visits within a seven-calendar-day inclusive window.
adopted=set(); first_adoption={}
for uid,g in e.groupby('user_id'):
 d=g.day.to_numpy(dtype='datetime64[D]')
 for i in range(2,len(d)):
  if (d[i]-d[i-2]).astype(int)<=6:
   adopted.add(uid);first_adoption[uid]=pd.Timestamp(d[i]);break
u['adopted']=u.object_id.isin(adopted).astype(int)
u['created']=pd.to_datetime(u.creation_time)
print('Total users:',len(u),'Adopted:',u.adopted.sum(),'Rate:',round(u.adopted.mean(),4))
print('Adoption by source:');print(u.groupby('creation_source').adopted.agg(['count','sum','mean']).sort_values('mean',ascending=False))
# Descriptive associations (not leakage-free forecasts)
for c in ['opted_in_to_mailing_list','enabled_for_marketing_drip']:
 print(c);print(u.groupby(c).adopted.agg(['count','mean']))
u['invited']=u.invited_by_user_id.notna().astype(int)
print('invited');print(u.groupby('invited').adopted.agg(['count','mean']))
# Prediction at signup: only signup-known attributes. 14-day maturity cutoff for labels,
# and outcome only counts adoption events within first 14 days.
maxday=e.day.max(); eligible=u[u.created.dt.normalize()<=maxday-pd.Timedelta(days=14)].copy()
first=e.groupby('user_id').day.min()
# Recalculate adoption for first 14 days after creation, preventing late-adoption leakage.
e2=e.merge(eligible[['object_id','created']],left_on='user_id',right_on='object_id')
e2=e2[(e2.day>=e2.created.dt.normalize())&(e2.day<e2.created.dt.normalize()+pd.Timedelta(days=14))]
yes=set()
for uid,g in e2.groupby('user_id'):
 d=g.day.to_numpy(dtype='datetime64[D]')
 if any((d[i]-d[i-2]).astype(int)<=6 for i in range(2,len(d))):yes.add(uid)
eligible['adopt_14d']=eligible.object_id.isin(yes).astype(int)
eligible['signup_month']=eligible.created.dt.month.astype(str)
eligible['signup_weekday']=eligible.created.dt.dayofweek.astype(str)
# Org size at signup cannot be reconstructed from current org membership; omit.
features=['creation_source','opted_in_to_mailing_list','enabled_for_marketing_drip','invited','signup_month','signup_weekday']
cat=['creation_source','signup_month','signup_weekday']; num=[x for x in features if x not in cat]
X=eligible[features]; y=eligible.adopt_14d
print('14-day eligible',len(y),'14-day adoption',y.sum(),round(y.mean(),4))
print('14-day by source');print(eligible.groupby('creation_source').adopt_14d.agg(['count','sum','mean']).sort_values('mean',ascending=False))
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num',StandardScaler(),num)])
model=Pipeline([('pre',pre),('lr',LogisticRegression(max_iter=1000,class_weight='balanced'))])
# Temporal split by signup date to simulate future-user prediction.
eligible=eligible.sort_values('created');cut=int(len(eligible)*.8)
train,test=eligible.iloc[:cut],eligible.iloc[cut:]
model.fit(train[features],train.adopt_14d)
p=model.predict_proba(test[features])[:,1]
print('Temporal holdout',len(test),'positive',test.adopt_14d.sum(),'ROC AUC',roc_auc_score(test.adopt_14d,p),'PR AUC',average_precision_score(test.adopt_14d,p),'baseline PR',test.adopt_14d.mean())
coefs=pd.DataFrame({'feature':model.named_steps['pre'].get_feature_names_out(),'coefficient':model.named_steps['lr'].coef_[0]});print(coefs.sort_values('coefficient',ascending=False).to_string(index=False))
u[['object_id','adopted']].to_csv(ROOT/'adoption_labels.csv',index=False)
u.groupby('creation_source').adopted.agg(users='count',adopted='sum',adoption_rate='mean').to_csv(ROOT/'adoption_by_source.csv')
coefs.to_csv(ROOT/'model_coefficients.csv',index=False)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=u.groupby('creation_source').adopted.mean().sort_values();fig,ax=plt.subplots(figsize=(9,4.5));s.mul(100).plot.barh(ax=ax);ax.set_xlabel('Ever adopted (%)');ax.set_ylabel('Signup source');ax.set_title('Adoption rate by account creation source');fig.tight_layout();fig.savefig(ROOT/'adoption_by_source.png',dpi=160);plt.close(fig)
