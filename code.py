import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option('display.max_columns', None)
train = pd.read_csv('train.gz',compression='gzip')
train.head()
train['hour']=train['hour'].apply(lambda x: x + 2000000000)
train['hour']=train['hour'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H').strftime('%Y-%m-%d-%H'))
train = train.rename(columns={"hour": "date"})
train.groupby('date')['id'].count().reset_index()

train = train[(train['date']>='2014-10-21-00')&(train['date']<='2014-10-21-23')]
train = pd.read_csv("finaltrain.csv")
train.shape
(4122995, 24)
train.dtypes
train['click'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='click',data=train, palette='hls')
plt.show();
train['click'].value_counts()/len(train)
train['hour_of_day'] = train['date'].apply(lambda x: int(x[-2:]))
train.groupby('hour_of_day').agg({'click':'sum'}).plot(figsize=(12,6))
plt.ylabel('Number of clicks')
plt.title('click trends by hour of day');
plt.show()
train.head(3)
train.groupby(['hour_of_day', 'click']).size().unstack().plot(kind='bar', title="Hour of Day", figsize=(12,6))
plt.ylabel('count')
plt.title('Hourly impressions vs. clicks');
plt.show()

import seaborn as sns

df_click = train[train['click'] == 1]
df_hour = train[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()
df_hour = df_hour.rename(columns={'click': 'impressions'})
df_hour['clicks'] = df_click[['hour_of_day','click']].groupby(['hour_of_day']).count().reset_index()['click']
df_hour['CTR'] = df_hour['clicks']/df_hour['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='hour_of_day', data=df_hour)
plt.title('Hourly CTR');
plt.show()

print(train.C1.value_counts()/len(train))

C1_values = train.C1.unique()
C1_values.sort()
ctr_avg_list=[]
for i in C1_values:
    ctr_avg=train.loc[np.where((train.C1 == i))].click.mean()
    ctr_avg_list.append(ctr_avg)
    print("{}: click through rate: {}".format(i,ctr_avg))
train.groupby(['C1', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='C1 histogram');
plt.show()
df_c1 = train[['C1','click']].groupby(['C1']).count().reset_index()
df_c1 = df_c1.rename(columns={'click': 'impressions'})
df_c1['clicks'] = df_click[['C1','click']].groupby(['C1']).count().reset_index()['click']
df_c1['CTR'] = df_c1['clicks']/df_c1['impressions']*100

plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='C1', data=df_c1)
plt.title('CTR by C1');
plt.show()

train['click'].mean()
df_c1.CTR.describe()
print(train.banner_pos.value_counts()/len(train))
banner_pos = train.banner_pos.unique()
banner_pos.sort()
ctr_avg_list=[]
for i in banner_pos:
    ctr_avg=train.loc[np.where((train.banner_pos == i))].click.mean()
    ctr_avg_list.append(ctr_avg)
    print("{}: click through rate: {}".format(i,ctr_avg))
train.groupby(['banner_pos', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='banner position histogram');
plt.show()
df_banner = train[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()
df_banner = df_banner.rename(columns={'click': 'impressions'})
df_banner['clicks'] = df_click[['banner_pos','click']].groupby(['banner_pos']).count().reset_index()['click']
df_banner['CTR'] = df_banner['clicks']/df_banner['impressions']*100
sort_banners = df_banner.sort_values(by='CTR',ascending=False)['banner_pos'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='banner_pos', data=df_banner, order=sort_banners)
plt.title('CTR by banner position');
plt.show()
df_banner.CTR.describe()

print("There are {} sites in the data set".format(train.site_id.nunique()))
print('The top 10 site ids that have the most impressions')
print((train.site_id.value_counts()/len(train))[0:10])
top10_ids = (train.site_id.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_ids:
    click_avg=train.loc[np.where((train.site_id == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site id value: {},  click through rate: {}".format(i,click_avg))
top10_sites = train[(train.site_id.isin((train.site_id.value_counts()/len(train))[0:10].index))]
top10_sites_click = top10_sites[top10_sites['click'] == 1]
top10_sites.groupby(['site_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site ids histogram');
plt.show()
df_site = top10_sites[['site_id','click']].groupby(['site_id']).count().reset_index()
df_site = df_site.rename(columns={'click': 'impressions'})
df_site['clicks'] = top10_sites_click[['site_id','click']].groupby(['site_id']).count().reset_index()['click']
df_site['CTR'] = df_site['clicks']/df_site['impressions']*100
sort_site = df_site.sort_values(by='CTR',ascending=False)['site_id'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_id', data=df_site, order=sort_site)
plt.title('CTR by top 10 site id');
plt.show()

print("There are {} site domains in the data set".format(train.site_domain.nunique()))
print('The top 10 site domains that have the most impressions')
print((train.site_domain.value_counts()/len(train))[0:10])
top10_domains = (train.site_domain.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_domains:
    click_avg=train.loc[np.where((train.site_domain == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site domain value: {},  click through rate: {}".format(i,click_avg))
top10_domain = train[(train.site_domain.isin((train.site_domain.value_counts()/len(train))[0:10].index))]
top10_domain_click = top10_domain[top10_domain['click'] == 1]
top10_domain.groupby(['site_domain', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site domains histogram');
plt.show()
df_domain = top10_domain[['site_domain','click']].groupby(['site_domain']).count().reset_index()
df_domain = df_domain.rename(columns={'click': 'impressions'})
df_domain['clicks'] = top10_domain_click[['site_domain','click']].groupby(['site_domain']).count().reset_index()['click']
df_domain['CTR'] = df_domain['clicks']/df_domain['impressions']*100
sort_domain = df_domain.sort_values(by='CTR',ascending=False)['site_domain'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_domain', data=df_domain, order=sort_domain)
plt.title('CTR by top 10 site domain');
plt.show()

print("There are {} site categories in the data set".format(train.site_category.nunique()))
print('The top 10 site categories that have the most impressions')
print((train.site_category.value_counts()/len(train))[0:10])
top10_categories = (train.site_category.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_categories:
    click_avg=train.loc[np.where((train.site_category == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for site category value: {},  click through rate: {}".format(i,click_avg))
top10_category = train[(train.site_category.isin((train.site_category.value_counts()/len(train))[0:10].index))]
top10_category_click = top10_category[top10_category['click'] == 1]
top10_category.groupby(['site_category', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 site categories histogram');
plt.show()
df_category = top10_category[['site_category','click']].groupby(['site_category']).count().reset_index()
df_category = df_category.rename(columns={'click': 'impressions'})
df_category['clicks'] = top10_category_click[['site_category','click']].groupby(['site_category']).count().reset_index()['click']
df_category['CTR'] = df_category['clicks']/df_category['impressions']*100
sort_category = df_category.sort_values(by='CTR',ascending=False)['site_category'].tolist()
plt.figure(figsize=(12,6))
sns.barplot(y='CTR', x='site_category', data=df_category, order=sort_category)
plt.title('CTR by top 10 site category');
plt.show()

print("There are {} devices in the data set".format(train.device_id.nunique()))
print('The top 10 devices that have the most impressions')
print((train.device_id.value_counts()/len(train))[0:10])
top10_devices = (train.device_id.value_counts()/len(train))[0:10].index
click_avg_list=[]

for i in top10_devices:
    click_avg=train.loc[np.where((train.device_id == i))].click.mean()
    click_avg_list.append(click_avg)
    print("for device id value: {},  click through rate: {}".format(i,click_avg))
top10_device = train[(train.device_id.isin((train.device_id.value_counts()/len(train))[0:10].index))]
top10_device_click = top10_device[top10_device['click'] == 1]
top10_device.groupby(['device_id', 'click']).size().unstack().plot(kind='bar', figsize=(12,6), title='Top 10 device ids histogram');
plt.show()


print("There are {} device ips in the data set".format(train.device_ip.nunique()))
print("There are {} device types in the data set".format(train.device_type.nunique()))
print("There are {} device models in the data set".format(train.device_model.nunique()))
print("There are {} device cnn types in the data set".format(train.device_conn_type.nunique()))
print('The impressions by device types')
print((train.device_type.value_counts()/len(train)))
train[['device_type','click']].groupby(['device_type','click']).size().unstack().plot(kind='bar', title='device types');
plt.show()

df_click[df_click['device_type']==1].groupby(['hour_of_day', 'click']).size().unstack().plot(kind='bar', title="Clicks from device type 1 by hour of day", figsize=(12,6));
plt.show()

device_type_click = df_click.groupby('device_type').agg({'click':'sum'}).reset_index()
device_type_impression = train.groupby('device_type').agg({'click':'count'}).reset_index().rename(columns={'click': 'impressions'})
merged_device_type = pd.merge(left = device_type_click , right = device_type_impression, how = 'inner', on = 'device_type')
merged_device_type['CTR'] = merged_device_type['click'] / merged_device_type['impressions']*100
merged_device_type

print("There are {} apps in the data set".format(train.app_id.nunique()))
print("There are {} app domains in the data set".format(train.app_domain.nunique()))
print("There are {} app categories in the data set".format(train.app_category.nunique()))


print('The impressions by app categories')
print((train.app_category.value_counts()/len(train)))
train['app_category'].value_counts().plot(kind='bar', title='App Category v/s Clicks')
train_app_category = train.groupby(['app_category', 'click']).size().unstack()
train_app_category.div(train_app_category.sum(axis=1), axis=0).plot(kind='bar', stacked=True, title="Intra-category CTR");
plt.show()

print("There are {} C14 in the data set".format(train.C14.nunique()))
print("There are {} C15 in the data set".format(train.C15.nunique()))
print("There are {} C16 in the data set".format(train.C16.nunique()))
print("There are {} C17 in the data set".format(train.C17.nunique()))
print("There are {} C18 in the data set".format(train.C18.nunique()))
print("There are {} C19 in the data set".format(train.C19.nunique()))
print("There are {} C20 in the data set".format(train.C20.nunique()))
train.groupby(['C15', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C15 distribution');
train.groupby(['C16', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C16 distribution');
train.groupby(['C18', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C18 distribution');
train.head(3)
def convert_obj_to_int(self):

    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]+new_col_suffix] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self
train = convert_obj_to_int(train)
train.head(3)

train.drop('id', axis=1, inplace=True)
train.drop('date_int', axis=1, inplace=True)
train.head()

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
x_train, x_test, y_train, y_test = train_test_split(train, train['click'], test_size=0.1, random_state=101)
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)
Predictions = logmodel.predict(x_test)
print(confusion_matrix(y_test, Predictions))

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
x_train_poly = poly.fit_transform(x_train)
poly_model = LogisticRegression()
poly_model.fit(x_train_poly, y_train)
Predictions_2 = poly_model.predict(x_test)
print(classification_report(y_test,Predictions_2))
print(confusion_matrix(y_test, Predictions_2))

class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'i':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})
ffm_train = FFMFormatPandas()
ffm_train_data = ffm_train.fit_transform(train, y='click')
X_train, X_test = train_test_split(ffm_train_data, test_size = 0.1, random_state = 5)
X_train.to_csv('x_train.txt', header=None, index=None, sep=' ', mode='a')
X_test.to_csv('x_test.txt', header=None, index=None, sep=' ', mode='a')

fm_model = xl.create_fm()
fm_model.setTrain('x_train.txt')
fm_model.setValidate('x_test.txt')
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
fm_model.fit(param, "./model.out")
fm_model.cv(param)
fm_model.setTest("x_test.txt")
fm_model.setSigmoid()
fm_model.predict("trained_models/model.out", "output/predictions.txt")
print(classification_report(y_test,Predictions))

ffm_model = xl.create_ffm()
ffm_model.setTrain('x_train.txt')
ffm_model.setValidate('x_test.txt')
param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
ffm_model.fit(param, "./model.out")
ffm_model.cv(param)
ffm_model.setTest("x_test.txt")
ffm_model.setSigmoid()
ffm_model.predict("trained_models/model.out", "output/predictions.txt")
print(classification_report(y_test,Predictions))

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
# Separate majority and minority classes
df_majority = train[train.click==0]
df_minority = train[train.click==1]
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=3404777,    # to match majority class
                                 random_state=123) # reproducible results
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.click.value_counts()


y = df_upsampled.click
X = df_upsampled.drop('click', axis=1)
clf_1 = LogisticRegression().fit(X, y)

pred_y_1 = clf_1.predict(X)
print(np.unique(pred_y_1))
print(accuracy_score(y, pred_y_1))
0.5652167528152358
print(classification_report(y, pred_y_1))

print(confusion_matrix(y, pred_y_1))

X_train, X_test, y_train, y_test = train_test_split(train, train['click'], test_size=0.25, random_state=27)
sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)
smote_pred = smote.predict(X_test)
accuracy_score(y_test, smote_pred)

print(classification_report(y_test, smote_pred))

print(confusion_matrix(y_test, smote_pred))

y = train.click
X = train.drop('click', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
X = pd.concat([X_train, y_train], axis=1)
not_click = X[X.click==0]
click = X[X.click==1]
not_click_downsampled = resample(not_click,
                                replace = False, # sample without replacement
                                n_samples = len(click), # match minority n
                                random_state = 27) # reproducible results
downsampled = pd.concat([not_click_downsampled, click])
downsampled.click.value_counts()

y_train = downsampled.click
X_train = downsampled.drop('click', axis=1)
undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)
undersampled_pred = undersampled.predict(X_test)
accuracy_score(y_test, undersampled_pred)

print(classification_report(y_test, undersampled_pred))


print(confusion_matrix(y_test, undersampled_pred))

test = pd.read_csv('test.gz',compression='gzip')

test.head()

test['hour']=test['hour'].apply(lambda x: x + 2000000000)

test['hour']=test['hour'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d%H').strftime('%Y-%m-%d-%H'))

test = test.rename(columns={"hour": "date"})

test.groupby('date')['id'].count().reset_index()


test.shape
test = convert_obj_to_int(test)
test.head(3)
test.drop('id', axis=1, inplace=True)
test.drop('date_int', axis=1, inplace=True)

test.head()

pred_y_1 = clf_1.predict(test)
print(np.unique(pred_y_1))

print(pred_y_1)

sampleSubmission = pd.read_csv('sampleSubmission.gz',compression='gzip')
sampleSubmission.head()

sampleSubmission['click'] = pred_y_1
sampleSubmission.to_csv('sampleSubmission.csv', mode = 'w', index=False)