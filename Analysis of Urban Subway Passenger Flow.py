import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from shapely.geometry import LineString

#读取线路数据
line = gpd.read_file('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/line.json', encoding = 'utf-8')

#读取站点数据
stop = pd.read_csv('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/stop.csv')
stop['stationnames'].drop_duplicates()

#地铁站点绘制
stop['geometry'] = gpd.points_from_xy(stop['lon'], stop['lat'])
stop = gpd.GeoDataFrame(stop)
# stop.plot()
# plt.show()

#将站点连接,形成轨道边
stop['linename1'] = stop['linename'].shift(-1)
stop['stationnames1'] = stop['stationnames'].shift(-1)
stop = stop[stop['linename1'] == stop['linename']]
# print(stop)

#站点重命名,站点名称前面加上该站点所属的线路名称
stop['line'] = stop['linename'].apply(lambda r:r.split('(')[0].lstrip('地铁'))
stop.loc[stop['line'] == '5号线支线', 'line'] = '5号线'
stop = stop.rename(columns = {'stationnames':'ostop', 'stationnames1':'dstop'})
stop['ostation'] = stop['line'] + stop['ostop']
stop['dstation'] = stop['line'] + stop['dstop']

#给每条轨道边附上权重
edge1 = stop[['ostation', 'dstation']]
edge1['duration'] = 3

#重新读取轨道站点数据
stop = pd.read_csv('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/stop.csv')
stop['line'] = stop['linename'].apply(lambda r:r.split('(')[0].lstrip('地铁'))
stop['station'] = stop['line'] + stop['stationnames']

#统计站点出现了多少次
tmp = stop.groupby('stationnames')['linename'].count().rename('count').reset_index()

#提取换乘站点
tmp = tmp[tmp['count'] > 2]
tmp = pd.merge(stop, tmp['stationnames'], on='stationnames')

#对同一个站点的不同线路间生成的数据形成两两对应表
tmp = tmp[['stationnames', 'line', 'station']].drop_duplicates()
tmp = pd.merge(tmp, tmp, on= 'stationnames')

#构建换成边的表
edge2 = tmp[tmp['line_x'] != tmp['line_y']][['station_x', 'station_y']]
edge2.columns = ['ostation', 'dstation']
edge2['duration'] = 5

#两表合并
edge = pd.concat([edge1, edge2])
node = list(edge['ostation'].drop_duplicates())

#构建网络
G = nx.Graph()
G.add_nodes_from(node)
G.add_weighted_edges_from(edge.values)
nx.draw(G, node_size = 20)

#读取数据
icdata = pd.read_csv("D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/icdata-sample.csv", header = None)
icdata.columns = ['cardid', 'date', 'time', 'station', 'mode', 'price', 'type']
metrodata = icdata[icdata['mode'] == '地铁']

#整理数据的线路站点名,保留有用的列,并存储OD数据
metrodata = metrodata.sort_values(by = ['cardid', 'date', 'time'])
for i in metrodata.columns:
    metrodata[i + '1'] = metrodata[i].shift(-1)
metrood = metrodata[(metrodata['cardid'] == metrodata['cardid1'])&(metrodata['price'] == 0)&(metrodata['price1'] > 0)]
metrood['oline'] = metrood['station'].apply(lambda r:r[:(r.find('线') + 1)])
metrood['ostation'] = metrood['station'].apply(lambda  r:r[(r.find('线') + 1):])
metrood['dline'] = metrood['station1'].apply(lambda r:r[:(r.find('线') + 1)])
metrood['dstation'] = metrood['station1'].apply(lambda  r:r[(r.find('线') + 1):])
metrood = metrood[['cardid', 'date', 'time', 'station', 'oline', 'ostation', 'time1', 'station1', 'dline', 'dstation']]
metrood.columns = ['cardid', 'date', 'otime', 'ostation', 'oline', 'ostop', 'dtime', 'dstation', 'dline', 'dstop']
#metrood.to_csv('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/metrood.csv')

#修改OD数据的站点名称,使其能够与GIS数据构建的网络中的站点名称对应
metrood.loc[metrood['ostop'] == '淞浜路', 'ostop'] = '淞滨路'
metrood.loc[metrood['dstop'] == '淞浜路', 'dstop'] = '淞滨路'
metrood.loc[metrood['ostop'] == '上海大学站', 'ostop'] = '上海大学'
metrood.loc[metrood['dstop'] == '上海大学站', 'dstop'] = '上海大学'
metrood.loc[metrood['ostop'] == '上海野生动物园', 'ostop'] = '野生动物园'
metrood.loc[metrood['dstop'] == '上海野生动物园', 'dstop'] = '野生动物园'
metrood.loc[metrood['ostop'] == '外高桥保税区北', 'ostop'] = '外高桥保税区北站'
metrood.loc[metrood['dstop'] == '外高桥保税区北', 'dstop'] = '外高桥保税区北站'
metrood.loc[metrood['ostop'] == '外高桥保税区南', 'ostop'] = '外高桥保税区南站'
metrood.loc[metrood['dstop'] == '外高桥保税区南', 'dstop'] = '外高桥保税区南站'
metrood.loc[metrood['ostop'] == '李子园路', 'ostop'] = '李子园'
metrood.loc[metrood['dstop'] == '李子园路', 'dstop'] = '李子园'
metrood['ostop'] = metrood['ostop'].str.lstrip(' ').str.rstrip(' ')
metrood['dstop'] = metrood['dstop'].str.lstrip(' ').str.rstrip(' ')
metrood['ostation'] = metrood['oline'] + metrood['ostop']
metrood['dstation'] = metrood['dline'] + metrood['dstop']

# #对OD信息进行去重(虽然我并不知道为什么会有重复的)
od_distinct = metrood[['ostation', 'dstation']].drop_duplicates()

# 提取出行路径
od_distinct['path'] = od_distinct.apply(
    lambda r:nx.shortest_path(G, source=r['ostation'],
                                target = r['dstation'],
                                weight= 'duration')
    , axis = 1)

#转换od_distinct
ls = []
for i in range(len(od_distinct)):
    r = od_distinct.iloc[i]
    #r['path']这个是一个series列,并不是一行,如果是r[['path']],那就是一行了,是一个DataFrame表格了
    tmp = pd.DataFrame(r['path'], columns=['o'])
    tmp['d'] = tmp['o'].shift(-1)
    tmp = tmp.iloc[:-1]
    tmp['ostation'] = r['ostation']
    tmp['dstation'] = r['dstation']
    ls.append(tmp)
od_path = pd.concat(ls)
#od_path.to_csv('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/od_path.csv', index = None, encoding = 'utf-8_sig')

#断面客流集计
metrood['Hour'] = metrood['otime'].apply(lambda r:r.split(':')[0])
tmp = pd.merge(metrood, od_path, on = ['ostation', 'dstation'])
metro_passenger = tmp.groupby(['o', 'd'])['cardid'].count().rename('count').reset_index()

#断面客流的分布绘制
linename = '2号线'
linestop = stop[stop['line'] == linename]
for i in linestop.columns:
    linestop[i + '1'] = linestop[i].shift(-1)
linestop = linestop[linestop['linename'] == linestop['linename1']]
linestop = linestop[['linename', 'line1', 'stationnames', 'stationnames1']]
linestop['o'] = linestop['line1'] + linestop['stationnames']
linestop['d'] = linestop['line1'] + linestop['stationnames1']
linestop = linestop[['o', 'd', 'stationnames', 'stationnames1', 'linename', 'line1']]


#匹配断面客流
linestop = pd.merge(metro_passenger, linestop, on = ['o', 'd'])

#提取上行客流
shangxing = linestop['linename'].drop_duplicates().iloc[0]
tmp = linestop[linestop['linename'] == shangxing]
tmp['x'] = range(len(tmp))

#提取下行客流
xiaxing = linestop['linename'].drop_duplicates().iloc[1]
tmp1 = linestop[linestop['linename'] == xiaxing]
tmp1['x'] = range(len(tmp1))
tmp1['x'] = len(tmp1) - tmp1['x'] - 1

#提取站点名称
stationnames = list(tmp['stationnames'])
stationnames.append(tmp['stationnames1'].iloc[-1])

tmp['count'] *= 25
tmp1['count'] *= 25

#断面客流柱状图绘制可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(1, (7, 4), dpi= 250)
ax1 = plt.subplot(111)
plt.bar(tmp['x'], tmp['count'], width= 0.4, label = shangxing)
plt.bar(tmp1['x'], -tmp1['count'], width= 0.4, label = xiaxing)
ax1.spines['bottom'].set_position(('data', 0))
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
plt.xticks(np.arange(len(stationnames)) - 0.5, stationnames, rotation = 90, size = 8)
plt.legend()
plt.ylabel('断面客流')
plt.xlabel('站点')
locs, labels = plt.yticks()
plt.yticks(locs, abs(locs.astype(int)))
Hour = metrood['Hour']
plt.title(linename + '' + '全天断面客流')
#plt.show()

#重新读取
stop = pd.read_csv('D:/Python练习/第7章-地铁IC刷卡-城市轨道交通客流分析/data/stop.csv')
stop['geometry'] = gpd.points_from_xy(stop['lon'], stop['lat'])
stop = gpd.GeoDataFrame(stop)
stop['line'] = stop['linename'].apply(lambda r:r.split('(')[0])
stop.loc[stop['line'] == '5号线支线', 'line'] = '5号线'

stop['ID'] = range(len(stop))
stop['ID'] = stop.groupby(['linename'])['ID'].rank().astype(int)
r = stop[stop['linename'] == '地铁4号线(内圈(宜山路-宜山路))'].iloc[0]
r['ID'] = len(stop[stop['linename'] == '地铁4号线(内圈(宜山路-宜山路))']) + 1
stop = stop._append(r)
r = stop[stop['linename'] == '地铁4号线(外圈(宜山路-宜山路))'].iloc[0]
r['ID'] = len(stop[stop['linename'] == '地铁4号线(外圈(宜山路-宜山路))']) + 1
stop = stop._append(r)
stop = stop.sort_values(by = ['linename', 'ID'])

r = line.iloc[0]
line_geometry = r['geometry']

tmp = stop[stop['linename'] == r['linename']]
for i in tmp.columns:
    tmp[i + '1'] = tmp[i].shift(-1)
tmp = tmp.iloc[:-1]
tmp = tmp[['stationnames', 'stationnames1', 'geometry', 'geometry1', 'linename']]

tmp['o_project'] = tmp['geometry'].apply(lambda r1:line_geometry.project(r1))
tmp['d_project'] = tmp['geometry1'].apply(lambda r1:line_geometry.project(r1))

def getline(r2, 表_geometry):
    ls = []
    if r2['o_project'] < r2['d_project']:
        tmp1 = np.linspace(r2['o_project'], r2['d_project'], 10)
    if r2['o_project'] > r2['d_project']:
        tmp1 = np.linspace(r2['o_project'] - line_geometry.length, r2['d_project'], 10)
        tmp1[tmp1 < 0] = tmp1[tmp1 < 0] + line_geometry.length
    for j in tmp1:
        ls.append(line_geometry.interpolate(j))
    return LineString(ls)
getline(tmp.iloc[0], line_geometry)

tmp['geometry'] = tmp.apply(lambda r2:getline(r2, line_geometry), axis = 1)
tmp = gpd.GeoDataFrame(tmp)
#tmp.plot(column = 'o_project')

lss = []
for k in range(len(line)):
    r = line.iloc[k]
    line_geometry = r['geometry']
    tmp = stop[stop['linename'] == r['linename']]
    for i in tmp.columns:
        tmp[i + '1'] = tmp[i].shift(-1)
    tmp = tmp.iloc[:-1]
    tmp = tmp[['stationnames', 'stationnames1', 'geometry', 'geometry1', 'linename']]
    tmp['o_project'] = tmp['geometry'].apply(lambda r1:r['geometry'].project(r1))
    tmp['d_project'] = tmp['geometry1'].apply(lambda r1: r['geometry'].project(r1))
    tmp['geometry'] = tmp.apply(lambda r2:getline(r2, line_geometry), axis = 1)
    lss._append(tmp)
metro_line_splited = pd.concat(lss)
metro_line_splited.plot(column = 'o_project')