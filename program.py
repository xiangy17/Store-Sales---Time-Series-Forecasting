train = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/train.csv')
test = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/test.csv')
oil = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/oil.csv')
stores = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/stores.csv')
transactions = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/transactions.csv')
holidays_events = pd.read_csv('/kaggle/input/competitions/store-sales-time-series-forecasting/holidays_events.csv')
#读取数据

print(train.shape)
print(train.head(5))

print(test.shape)
print(test.head(5))

print(oil.shape)
print(oil.head(5))

print(stores.shape)
print(stores.head(5))

print(transactions.shape)
print(transactions.tail(5))

print(holidays_events.shape)
print(holidays_events.head(5))
#检查数据格式

import matplotlib.pyplot as plt
import seaborn as sns

def draw(x,y):
    fig = plt.figure(figsize=(8,8))
    plt.plot(x,y)

draw(oil['date'],oil['dcoilwtico'])
#油价分析

oil = oil.bfill()
#油价按下一个填充，第一个值缺少

def add_feature():
    holidays_events.rename(columns={'date': 'date',
                                       'type': 'Daily_holiday_type',
                                       'locale': 'Daily_holiday_locale',
                                       'locale_name': 'Daily_holiday_locale_name',
                                       'description': "Daily_holiday_description",
                                       'transferred': "Daily_holiday_transferred"},inplace=True)
    stores.rename(columns={'store_nbr': 'store_nbr',
                              'city': 'stores_city',
                              'state': 'store_state',
                              'type': 'store_type',
                              'cluster': 'store_cluster'},
                     inplace=True)
    transactions.rename(columns={'transactions': 'Daily_transactions'},inplace=True)

    train_n = pd.merge(train,holidays_events,how='left',left_on='date',right_on='date')
    test_n = pd.merge(test, holidays_events, how='left', left_on='date', right_on='date')
    train_n = pd.merge(train_n,oil,how='left', left_on='date', right_on='date')
    test_n = pd.merge(test_n,oil,how='left', left_on='date', right_on='date')
    train_n = pd.merge(train_n,stores,how='left', left_on='store_nbr', right_on='store_nbr')
    test_n = pd.merge(test_n,stores,how='left', left_on='store_nbr', right_on='store_nbr')
    return train_n,test_n

res = add_feature()
train_n = res[0]
test_n = res[1]
#合并特征

def look(a):
    print(f'length:{len(a)}')
    for i in a.columns:
        b = a[i].describe()
        print(f'{i}: rate:{b['count']/len(a[i])}')

print('train_n>>>>>>>>')
look(train_n)
print('test_n>>>>>>>>>')
look(test_n)
print('test>>>>>>>>>>>')
look(test)
#缺失值统计

train_n.drop('Daily_transactions',axis=1,inplace=True)
test_n.drop('Daily_transactions',axis=1,inplace=True)
#这个特征test中没有，为保证一致性去除

train_n['dcoilwtico'].ffill(inplace=True)
test_n['dcoilwtico'].ffill(inplace=True)
#merge中只对first的油价合并，而日期是一定的，故按上一个值填充

train_n.fillna('no',inplace=True)
test_n.fillna('no',inplace=True)
#针对缺失的节假日填充为no

train_n.isnull().sum()
test_n.isnull().sum()
#检查无缺失值

train_n.shape
#这里train的行数比原来的多了，原因是一天可以对应多个节假日，保持特征不变，通过合并与否检验，发现不处理的mae更好

def prework(x):
    x['date'] = x['date'].apply(lambda a:int(str(a).split('-')[0]+str(a).split('-')[1] + str(a).split('-')[2]))
    x['Daily_holiday_type'] = pd.factorize(x['Daily_holiday_type'])[0].astype(int)
    x['Daily_holiday_locale'] = pd.factorize(x['Daily_holiday_locale'])[0].astype(int)
    x['Daily_holiday_locale_name'] = pd.factorize(x['Daily_holiday_locale_name'])[0].astype(int)
    x['Daily_holiday_description'] = pd.factorize(x['Daily_holiday_description'])[0].astype(int)
    x['Daily_holiday_transferred'] = pd.factorize(x['Daily_holiday_transferred'])[0].astype(int)
    x['stores_city'] = pd.factorize(x['stores_city'])[0].astype(int)
    x['store_state'] = pd.factorize(x['store_state'])[0].astype(int)
    x['store_type'] = pd.factorize(x['store_type'])[0].astype(int)

prework(train_n)
prework(test_n)
#数据预处理，用factorize不用dummie，unique特征太多

train_n['family'] = pd.factorize(train_n['family'])[0].astype(int)
test_n['family'] = pd.factorize(test_n['family'])[0].astype(int)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def decisiontree(xf,yf,data):
    x_ = data[xf]
    y_ = data[yf]
    model = DecisionTreeRegressor(random_state=17,max_depth=100)
    a_x,b_x,a_y,b_y = train_test_split(x_,y_,random_state=17)
    model.fit(a_x,a_y)
    predictions = model.predict(b_x)
    delta = mean_absolute_error(b_y,predictions)
    print(f'decisiontree mae:{delta}')
#决策树


def randomforest(xf,yf,data):
    x_ = data[xf]
    y_ = data[yf]
    model = RandomForestRegressor(random_state=17,max_depth=100)
    a_x,b_x,a_y,b_y = train_test_split(x_,y_,random_state=17)
    model.fit(a_x,a_y)
    predictions = model.predict(b_x)
    delta = mean_absolute_error(b_y,predictions)
    print(f'randomforest mae:{delta}')
#随机森林

X_Feature=['id', 'date', 'store_nbr', 'family', 'onpromotion',
             'Daily_holiday_type', 'Daily_holiday_locale',
             'Daily_holiday_locale_name', 'Daily_holiday_description',
             'Daily_holiday_transferred', 'dcoilwtico', 'stores_city', 'store_state',
             'store_type', 'store_cluster',]
Y_Feature = 'sales'

decisiontree(X_Feature, Y_Feature, train_n)
randomforest(X_Feature, Y_Feature, train_n)
#随机森林效果好点


def forest():
    model = RandomForestRegressor(random_state=1,max_depth=100)
    aim = test_n[X_Feature]
    predictions = model.predict(aim)
    sub = pd.DataFrame()
    sub['id'] = test_n['id']
    sub['sales'] = predictions
    sub.to_csv('submission.csv',index=False)

#提交

