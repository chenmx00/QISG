import pymysql


def connect():
    return pymysql.connect(
        user='root', password='', database='spec', charset='utf8')