import xml.etree.cElementTree as et
import pandas as pd
import datetime, time


def format1(file_path):
    '''综自数据格式化'''
    # 读取xml文件，放到dataframe df_xml中
    print('加载文件')
    xml_tree = et.ElementTree(
        file=file_path)  # 文件路径
    dfcols = ['C0', 'ID', 'DATA', 'CREATE_TIME', 'VALUE', 'UNIT']
    df_xml = pd.DataFrame(columns=dfcols)
    root = xml_tree.getroot()
    print('root', len(root))

    row_num = 0
    for row_node in root:
        row_num += 1
        print('\rrow_node %d/%d' % (row_num, len(root)), end='')
        C0 = ''
        ID = ''
        DATA = ''
        CREATE_TIME = ''
        VALUE = ''
        UNIT = ''
        for node in row_node:
            # print(node, node.tag, node.attrib, node.text)
            if node.tag == 'C0':
                C0 = node.text
            if node.tag == 'ID':
                ID = node.text
            if node.tag == 'DATA':
                DATA = node.text
            if node.tag == 'CREATE_TIME':
                CREATE_TIME = node.text
            if node.tag == 'VALUE':
                VALUE = node.text
            if node.tag == 'UNIT':
                UNIT = node.text

        # print(C0, ID, DATA, CREATE_TIME, VALUE, UNIT)

        df_xml = df_xml.append(
            pd.Series([C0, ID, DATA, CREATE_TIME, VALUE, UNIT], index=dfcols),
            ignore_index=True)

    df_xml.loc[:, 'CREATE_TIME'] = df_xml.apply(
        lambda x: (datetime.datetime.strptime(x['CREATE_TIME'].replace('月 ', '').replace('月', '')
                                              .replace('下午', 'PM').replace('上午', 'AM'),
                                              '%d-%m-%y %I.%M.%S.%f %p')).strftime('%Y/%m/%d %H:%M:%S'), axis=1)
    df_xml.to_csv('./data/bdzdata1.csv')


def format2(file_path):
    '''红外数据格式化'''
    # 读取xml文件，放到dataframe df_xml中
    print('加载文件')
    xml_tree = et.ElementTree(
        file=file_path)  # 文件路径
    dfcols = ['C0', 'ID', 'PRE_POINT_NAME',
              'AREA_NAME', 'CREATE_TIME', 'VALUE', 'UNIT']
    df_xml = pd.DataFrame(columns=dfcols)
    root = xml_tree.getroot()
    print('root', len(root))

    row_num = 0
    for row_node in root:
        row_num += 1
        print('\rrow_node %d/%d' % (row_num, len(root)), end='')
        C0 = ''
        ID = ''
        PRE_POINT_NAME = ''
        AREA_NAME = ''
        CREATE_TIME = ''
        VALUE = ''
        UNIT = ''
        for node in row_node:
            # print(node, node.tag, node.attrib, node.text)
            if node.tag == 'C0':
                C0 = node.text
            if node.tag == 'ID':
                ID = node.text
            if node.tag == 'PRE_POINT_NAME':
                PRE_POINT_NAME = node.text
            if node.tag == 'AREA_NAME':
                AREA_NAME = node.text
            if node.tag == 'CREATE_TIME':
                CREATE_TIME = node.text
            if node.tag == 'VALUE':
                VALUE = node.text
            if node.tag == 'UNIT':
                UNIT = node.text

        # print(C0, ID, DATA, CREATE_TIME, VALUE, UNIT)

        df_xml = df_xml.append(
            pd.Series([C0, ID, PRE_POINT_NAME, AREA_NAME,
                       CREATE_TIME, VALUE, UNIT], index=dfcols),
            ignore_index=True)

    df_xml.loc[:, 'CREATE_TIME'] = df_xml.apply(
        lambda x: (datetime.datetime.strptime(x['CREATE_TIME'].replace('月 ', '').replace('月', '')
                                              .replace('下午', 'PM').replace('上午', 'AM'),
                                              '%d-%m-%y %I.%M.%S.%f %p')).strftime('%Y/%m/%d %H:%M:%S'), axis=1)
    df_xml.to_csv('./data/bdzdata2.csv')


def getOneZongZhiData(file_path, data_name):
    '''读取单个传感器的综自数据'''
    df = pd.read_csv(file_path)
    df = df.loc[(df['DATA'] == data_name)]
    # # 去掉秒数
    # df.loc[:, 'CREATE_TIME'] = df.apply(
    #     lambda x: (datetime.datetime.strptime(x['CREATE_TIME'], '%Y/%m/%d %H:%M:%S')
    #                .strftime('%Y/%m/%d %H:%M:00')), axis=1)
    return df.loc[:, ['CREATE_TIME','VALUE']].copy()


def getOneHotData(file_path, pre_point_name, area_name):
    '''读取单个个红外测温区数据'''
    df = pd.read_csv(file_path)
    df = df.loc[(df['PRE_POINT_NAME'] == pre_point_name)
                & (df['AREA_NAME'] == area_name)]
    # # 去掉秒数
    # df.loc[:, 'CREATE_TIME'] = df.apply(
    #     lambda x: (datetime.datetime.strptime(x['CREATE_TIME'], '%Y/%m/%d %H:%M:%S')
    #                .strftime('%Y/%m/%d %H:%M:00')), axis=1)
    return df.loc[:, ['CREATE_TIME','VALUE']].copy()


# print(datetime.datetime.strptime('20-6月 -19 04.00.00.759000 下午'.replace('月 ', '').replace('月', '')
#         .replace('下午', 'PM').replace('上午', 'AM'),
#         '%d-%m-%y %I.%M.%S.%f %p').strftime('%Y/%m/%d %H:%M:%S'))

# format1('E:\Labels\数据预测\数据导出(镇坪19111512-19112412)\动环监测数据.xml')
# format2('E:\Labels\数据预测\数据导出(镇坪19111512-19112412)\红外数据.xml')

# df1 = getOneHotData('./data/bdzdata2.csv', '#1主变201A T型线夹', '#1主变201A T型线夹')
# print(df1)
# df2 = getOneHotData('./data/bdzdata2.csv', '#1主变201B线夹', '#1主变201B线夹')
# print(df2)
# df3 = getOneHotData('./data/bdzdata2.csv', '#1主变油枕刻度表', '#1主变油枕刻度表')
# print(df3)
df1 = getOneZongZhiData('./data/bdzdata1.csv', '1B主变高压侧A相电流')
print(df1)
df2 = getOneZongZhiData('./data/bdzdata1.csv', '211馈线电流')
print(df2)
df3 = getOneZongZhiData('./data/bdzdata1.csv', '211馈线功率因数')
print(df3)
df4 = getOneZongZhiData('./data/bdzdata1.csv', '211馈线无功')
print(df4)
df5 = getOneZongZhiData('./data/bdzdata1.csv', '211馈线有功')
print(df5)
df_merge = pd.merge(df1, df2, how='outer', on='CREATE_TIME', sort=False,
                    suffixes=('', '_df2'))
df_merge = pd.merge(df_merge, df3, how='outer', on='CREATE_TIME', sort=False,
                    suffixes=('', '_df3'))
df_merge = pd.merge(df_merge, df4, how='outer', on='CREATE_TIME', sort=False,
                    suffixes=('', '_df4'))
df_merge = pd.merge(df_merge, df5, how='outer', on='CREATE_TIME', sort=False,
                    suffixes=('', '_df5'))
# 排序
df_merge = df_merge.sort_values(by="CREATE_TIME",ascending=True)

# 增加列，作为索引
col_name = df_merge.columns.tolist()
col_name.insert(1, 'index')  # 默认值为NaN
df_merge = df_merge.reindex(columns=col_name)
# 用时间戳作为索引排序
df_merge.loc[:, 'index'] = df_merge.apply(lambda x: (int(time.mktime(datetime.datetime.strptime(x['CREATE_TIME'],'%Y/%m/%d %H:%M:%S').timetuple()))), axis=1)
df_merge = df_merge.sort_values(by="index",ascending=True)
df_merge = df_merge.set_index('index')
# 找最近值填充NaN
df_merge = df_merge.interpolate(method='nearest', axis=0, limit_direction='both')
# 去掉秒数
df_merge.loc[:, 'CREATE_TIME'] = df_merge.apply(
    lambda x: (datetime.datetime.strptime(x['CREATE_TIME'], '%Y/%m/%d %H:%M:%S')
                .strftime('%Y/%m/%d %H:%M:00')), axis=1)
# 去重
df_merge = df_merge.drop_duplicates()
# 去掉前后空值
df_merge = df_merge.dropna()
print(df_merge)
df_merge.to_csv('./data/bdzdata_merge.csv')