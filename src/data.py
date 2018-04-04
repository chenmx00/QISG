import numpy as np
import tushare as ts
import pymysql as sql
import db
import pandas as pd
import datetime
import random

TIMESTEPS = 15
SAMPLES = 50000
FEATURES = 5  # 5/15/30/60/日
STEPS = [1, 3, 6, 12, 48]  # 5/15/30/60/日
PREV_MIN = 48 * TIMESTEPS  # 需要TIMESTEPS个日线数据，一个日线数据对应48个5分钟


def datetime_to_5min(d):
    if d.second > 0:
        d += datetime.timedelta(minutes=1)
        d = d.replace(second=0)
    om = d.minute % 5
    if om > 0:
        d += datetime.timedelta(minutes=5 - om)
    return d


def price(df, index):
    return float(df.iloc[index].price)


def sample(df):
    s = np.ndarray([FEATURES, TIMESTEPS])
    return s


def make():

    ds = []
    dl = []

    sqlconn = db.connect()
    cursor = sqlconn.cursor()

    cursor.execute("select code, count(code) from stock_price group by code")

    code_data_count = cursor.fetchall()

    s = 0
    c = 0

    for code, count in code_data_count:
        s += count

    for code, count in code_data_count:
        sample_count = int(count / s * SAMPLES + 1)

        df = pd.read_sql(
            "select time, price from stock_price where code=%s order by time asc",
            sqlconn,
            params=[code])

        for i_sample in range(sample_count):
            i_time = random.randint(PREV_MIN,
                                    df.shape[0] - 48 - 1)  # 末尾也留出一个日线数据

            sample = np.ndarray([FEATURES, TIMESTEPS])
            label = np.ndarray([FEATURES])

            price_baseline = price(df, i_time)

            for i_step in range(5):

                for i in range(1, TIMESTEPS + 1):
                    sample[i_step, TIMESTEPS - i] = price(
                        df, i_time - i * STEPS[i_step])
                price_next = price(df, i_time + STEPS[i_step])

                if price_next > price_baseline:
                    label[i_step] = 1
                elif price_next < price_baseline:
                    label[i_step] = -1
                else:
                    label[i_step] = 0

            sample -= price_baseline
            sample /= price_baseline

            ds.append(sample)
            dl.append(label)

            c += 1
            if c % 10 == 0:
                print("采样{0}/{1}".format(c, SAMPLES))

    ds = np.asarray(ds)
    dl = np.asarray(dl)

    np.savez("data\stock.npz", price=ds, label=dl)
    print("done")


if __name__ == "__main__":
    make()