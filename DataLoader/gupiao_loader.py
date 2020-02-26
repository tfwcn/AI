import tushare as ts
import os
import pandas
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import mpl_finance as mpf
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class GuPiaoLoader():
    '''加载股票文件'''

    def __init__(self):
        with open('./data/gupiao_token.txt', 'r', encoding='UTF-8') as txt_file:
            tmp_token = txt_file.readline()
            print('token:', tmp_token)
            self.set_token(tmp_token)

    def set_token(self, api_token):
        '''设置API Token'''
        ts.set_token(api_token)
        self.pro = ts.pro_api()

    def download_code_list(self, data_dir):
        '''下载股票列表'''
        # 创建目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # df = pro.daily(ts_code='603106.SH')
        # 获取所有股票代码
        data = self.pro.stock_basic(list_status='L')
        code_list = data['ts_code'].values
        print('code_list', len(code_list), code_list)
        data.to_csv(os.path.join(data_dir, 'data.csv'))
        return code_list

    def download_all(self, data_dir):
        '''下载股票数据'''
        # 创建目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # 获取所有股票代码
        code_list = download_code_list(data_dir)
        for now_code in code_list:
            self.download_one(data_dir, now_code)

    def download_one(self, data_dir, now_code):
        '''下载股票数据'''
        # 创建目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        file_name = os.path.join(data_dir, '%s.csv' % (now_code))
        print(file_name)
        # 下载某只股票数据
        df = ts.pro_bar(ts_code=now_code, adj='qfq')
        # print(df)
        df.to_csv(file_name)
        print('已导出%s' % (now_code))

    def convert_file(self, data_dir):
        '''转换导出数据，通过其他股票软件导出'''
        for root, dirs, files in os.walk(data_dir):
            # 遍历文件
            for f in files:
                if f[-4:] == '.csv':
                    file_name = os.path.join(root, f)
                    print('加载文件', file_name)
                    with open(file_name, 'r') as r:
                        lines = r.readlines()
                    if len(lines) == 0:
                        continue
                    print(lines[0][:6], os.path.basename(file_name)[:6])
                    if lines[0][:6] != os.path.basename(file_name)[:6]:
                        continue
                    lines[1] = lines[1].replace('\t', ',').replace(' ', '')
                    print('lines[1]', lines[1])
                    with open(file_name, 'w', encoding='utf-8') as w:
                        w.writelines(lines[1:-1])

    def load_one(self, file_name):
        '''加载数据文件，并把价格转换成升降比例'''
        print('加载文件', file_name)
        df = pandas.read_csv(file_name)
        # 删除NaN值
        df = df.dropna()
        # 增加列
        col_name = df.columns.tolist()
        col_name.append('开盘%')  # 默认值为NaN
        col_name.append('最高%')  # 默认值为NaN
        col_name.append('最低%')  # 默认值为NaN
        col_name.append('收盘%')  # 默认值为NaN
        df = df.reindex(columns=col_name)
        # 填充NaN值为0
        df = df.fillna(value=0.0)
        # 第一条记录，用开盘价计算
        old_plice = df.loc[0, '开盘']
        df.loc[0, '开盘%'] = df.loc[0, '开盘']/old_plice-1
        df.loc[0, '最高%'] = df.loc[0, '最高']/old_plice-1
        df.loc[0, '最低%'] = df.loc[0, '最低']/old_plice-1
        df.loc[0, '收盘%'] = df.loc[0, '收盘']/old_plice-1
        for i in range(1, len(df)):
            old_plice = df.loc[i-1, '收盘']
            df.loc[i, '开盘%'] = df.loc[i, '开盘']/old_plice-1
            df.loc[i, '最高%'] = df.loc[i, '最高']/old_plice-1
            df.loc[i, '最低%'] = df.loc[i, '最低']/old_plice-1
            df.loc[i, '收盘%'] = df.loc[i, '收盘']/old_plice-1
        return df

    def get_random_data(self, df_sh, df_sz, df_target, history_size, target_size, start_index=None):
        '''根据数据窗口获取数据'''
        data = []
        labels = []
        # 日期同步
        tmp_df_sh = df_sh.loc[df_sh['日期'] >= df_target.loc[0,'日期'],
                             ['日期', '开盘%', '最高%', '最低%', '收盘%']]
        # print('tmp_df_sh',tmp_df_sh)
        tmp_df_sz = df_sz.loc[df_sz['日期'] >= df_target.loc[0,'日期'],
                             ['日期', '开盘%', '最高%', '最低%', '收盘%']]
        # print('tmp_df_sz',tmp_df_sz)
        tmp_df_target = df_target.loc[:, ['日期', '开盘%', '最高%', '最低%', '收盘%']]
        # print('tmp_df_target',tmp_df_target)
        # 随机取一段时间数据
        if start_index==None:
            start_index = random.randint(history_size, len(tmp_df_sh)-target_size-1)
        tmp_df_target = tmp_df_target.iloc[start_index-history_size:start_index+target_size]
        # 数据归一化
        tmp_df_sh.loc[:, '开盘%'] = tmp_df_sh.apply(lambda x: x['开盘%'] * 10, axis=1)
        tmp_df_sh.loc[:, '最高%'] = tmp_df_sh.apply(lambda x: x['最高%'] * 10, axis=1)
        tmp_df_sh.loc[:, '最低%'] = tmp_df_sh.apply(lambda x: x['最低%'] * 10, axis=1)
        tmp_df_sh.loc[:, '收盘%'] = tmp_df_sh.apply(lambda x: x['收盘%'] * 10, axis=1)
        tmp_df_sz.loc[:, '开盘%'] = tmp_df_sz.apply(lambda x: x['开盘%'] * 10, axis=1)
        tmp_df_sz.loc[:, '最高%'] = tmp_df_sz.apply(lambda x: x['最高%'] * 10, axis=1)
        tmp_df_sz.loc[:, '最低%'] = tmp_df_sz.apply(lambda x: x['最低%'] * 10, axis=1)
        tmp_df_sz.loc[:, '收盘%'] = tmp_df_sz.apply(lambda x: x['收盘%'] * 10, axis=1)
        tmp_df_target.loc[:, '开盘%'] = tmp_df_target.apply(lambda x: x['开盘%'] * 10, axis=1)
        tmp_df_target.loc[:, '最高%'] = tmp_df_target.apply(lambda x: x['最高%'] * 10, axis=1)
        tmp_df_target.loc[:, '最低%'] = tmp_df_target.apply(lambda x: x['最低%'] * 10, axis=1)
        tmp_df_target.loc[:, '收盘%'] = tmp_df_target.apply(lambda x: x['收盘%'] * 10, axis=1)
        # 合并数据
        tmp_df_merge = pandas.merge(tmp_df_target, tmp_df_sh, how='left', on='日期', sort=False,
                                    suffixes=('_target', '_sh'))
        tmp_df_merge = pandas.merge(
            tmp_df_merge, tmp_df_sz, how='left', on='日期', sort=False)
        # 删除NaN值
        tmp_df_merge = tmp_df_merge.dropna()
        # 增加列
        col_name = tmp_df_merge.columns.tolist()
        col_name.insert(1, '年')  # 默认值为NaN
        col_name.insert(2, '月')  # 默认值为NaN
        col_name.insert(3, '日')  # 默认值为NaN
        tmp_df_merge = tmp_df_merge.reindex(columns=col_name)
        # 日期数据归一化
        tmp_df_merge.loc[:, '年'] = tmp_df_merge.apply(lambda x: (datetime.datetime.strptime(x['日期'],'%Y/%m/%d').year-2000) / 20, axis=1)
        tmp_df_merge.loc[:, '月'] = tmp_df_merge.apply(lambda x: (datetime.datetime.strptime(x['日期'],'%Y/%m/%d').month) / 12, axis=1)
        tmp_df_merge.loc[:, '日'] = tmp_df_merge.apply(lambda x: (datetime.datetime.strptime(x['日期'],'%Y/%m/%d').day) / 31, axis=1)
        # print('tmp_df_merge', len(tmp_df_merge))
        # print(tmp_df_merge.loc[:, ['日期','年','月','日']])
        # print(tmp_df_merge)
        return tmp_df_merge

    def get_data_to_train(self, df_sh, df_sz, df_target, batch_size, history_size, target_size, start_index=None):
        '''数据格式化用于训练
        batch_size:批次大小
        history_size:训练数据大小
        target_size:预测数据大小
        '''
        x = []
        y = []
        for _ in range(batch_size):
            tmp_df = self.get_random_data(df_sh, df_sz, df_target, history_size, target_size, start_index)
            tmp_values = tmp_df.values[:,1:]
            # print('tmp_values', tmp_values.shape)
            x.append(tmp_values[:history_size,:].tolist())
            y.append(tmp_values[history_size:history_size+target_size,:].tolist())
        x = np.array(x)
        y = np.array(y)
        # print('x', x.shape, x)
        # print('y', y.shape, y)
        return x, y

    def get_data_to_predict(self, df_sh, df_sz, df_target, history_size, target_size, start_index=None):
        '''数据格式化用于训练
        batch_size:批次大小
        history_size:训练数据大小
        target_size:预测数据大小
        '''
        if start_index==None:
            start_index = len(df_target)
        tmp_df = self.get_random_data(df_sh, df_sz, df_target, history_size, 0, start_index)
        # print(tmp_df)
        # 排除日期列
        tmp_values = tmp_df.values[:,1:]
        # print('tmp_values', tmp_values.shape)
        x = tmp_values[:history_size,:]
        x = np.expand_dims(x, axis=0)
        time_step = self.create_time(tmp_df.iloc[history_size-1,:].loc['日期'], target_size)
        time_step = np.expand_dims(time_step, axis=0)
        return x, time_step
    
    def data_generator(self, df_sh, df_sz, df_target, batch_size, history_size, target_size):
        '''循环生成数据'''
        while True:
            x, y = self.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
            # print('x', x.shape, 'y', y.shape)
            yield x, y

    def create_time(self, start_time, target_size):
        '''
        创建预测时序
        target_size:预测数据大小
        '''
        tmp_start_time = datetime.datetime.strptime(start_time,'%Y/%m/%d')
        result = []
        for i in range(target_size):
            if tmp_start_time.weekday==4:
                tmp_start_time = tmp_start_time + datetime.timedelta(days=3)
            else:
                tmp_start_time = tmp_start_time + datetime.timedelta(days=1)
            tmp_year = (tmp_start_time.year - 2000) / 20
            tmp_month = tmp_start_time.month / 12
            tmp_day = tmp_start_time.day / 31
            result.append([tmp_year, tmp_month, tmp_day])
        
        result = np.array(result)
        return result

    def show_image(self, history_data, target_data=None, true_data=None):
        '''
        显示K线图
        history_data:(None,15)
        target_data:(None,15)
        '''
        all_data = history_data
        all_data2 = history_data
        if target_data is not None:
            all_data = np.append(history_data, target_data, axis=0)
        if true_data is not None:
            all_data2 = np.append(history_data, true_data, axis=0)
        show_history_data = pandas.DataFrame({'data':[i for i in range(all_data.shape[0])],
                                            'open':all_data[:,-12],
                                            'high':all_data[:,-11],
                                            'low':all_data[:,-10],
                                            'close':all_data[:,-9]})
        
        if true_data is not None:
            show_history_data2 = pandas.DataFrame({'data':[i for i in range(all_data2.shape[0])],
                                                'open':all_data2[:,-12],
                                                'high':all_data2[:,-11],
                                                'low':all_data2[:,-10],
                                                'close':all_data2[:,-9]})
        # print('show_history_data', show_history_data)
        now_close = 50
        for i in range(len(show_history_data)):
            show_history_data.loc[i,'open'] = now_close*(1+show_history_data.loc[i,'open']*0.1)
            show_history_data.loc[i,'high'] = now_close*(1+show_history_data.loc[i,'high']*0.1)
            show_history_data.loc[i,'low'] = now_close*(1+show_history_data.loc[i,'low']*0.1)
            now_close = now_close*(1+show_history_data.loc[i,'close']*0.1)
            show_history_data.loc[i,'close'] = now_close

        if true_data is not None:
            now_close = 50
            for i in range(len(show_history_data2)):
                show_history_data2.loc[i,'open'] = now_close*(1+show_history_data2.loc[i,'open']*0.1)
                show_history_data2.loc[i,'high'] = now_close*(1+show_history_data2.loc[i,'high']*0.1)
                show_history_data2.loc[i,'low'] = now_close*(1+show_history_data2.loc[i,'low']*0.1)
                now_close = now_close*(1+show_history_data2.loc[i,'close']*0.1)
                show_history_data2.loc[i,'close'] = now_close

        # print(show_history_data)
        # 创建一个子图 
        fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))
        fig.subplots_adjust(bottom=0.2)
        # 设置X轴刻度为日期时间
        # ax.xaxis_date()
        # # X轴刻度文字倾斜45度
        # plt.xticks(rotation=45)
        plt.title("股票K线图")
        plt.xlabel("时间")
        plt.ylabel("股价变化(%)")
        all_values = show_history_data.values
        all_values2 = show_history_data2.values
        if target_data is not None:
            mpf.candlestick_ohlc(ax,all_values[:len(history_data)],width=0.5,colorup='r',colordown='g')
            mpf.candlestick_ohlc(ax,all_values[len(history_data):],width=0.5,colorup='y',colordown='b')
        else:
            mpf.candlestick_ohlc(ax,all_values,width=0.5,colorup='r',colordown='g')
        
        if true_data is not None:
            mpf.candlestick_ohlc(ax,all_values2[len(history_data):],width=0.5,colorup='r',colordown='g')
        plt.show()

def main():
    gupiao_loader = GuPiaoLoader()
    gupiao_loader.download_code_list('./data/gupiao_data')
    # gupiao_loader.download_one('./data/gupiao_data', '000001.SH')
    # gupiao_loader.download_all('./data/gupiao_data')
    # gupiao_loader.convert_file('./data/gupiao_data')
    # df_sh = gupiao_loader.load_one('./data/gupiao_data/999999.SH.csv')
    # df_sz = gupiao_loader.load_one('./data/gupiao_data/399001.SZ.csv')
    # df_target = gupiao_loader.load_one('./data/gupiao_data/603106.SH.csv')
    # # print('df_target', df_target[df_target['日期']>='2017/01/01'])
    # batch_size = 20
    # history_size = 30
    # target_size = 5
    # x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    # print(x.shape,y.shape)
    # x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    # print(x.shape,y.shape)
    # x, y = gupiao_loader.get_data_to_train(df_sh, df_sz, df_target, batch_size, history_size, target_size)
    # print(x.shape,y.shape)
    # x, time_step = gupiao_loader.get_data_to_predict(df_sh, df_sz, df_target, history_size, target_size)
    # print(x.shape)
    # # 预测输入数据显示
    # gupiao_loader.show_image(x[0,:,:], y[0,:,:])
    # # 预测结果数据显示
    # gupiao_loader.show_image(y[0,:,:])


if __name__ == '__main__':
    main()
