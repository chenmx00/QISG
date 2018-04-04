import demjson
import requests
import pymysql


def refresh():

    qh_namesAll = [
        'pta_qh', 'czy_qh', 'ycz_qh', 'czp_qh', 'dlm_qh', 'qm_qh', 'jdm_qh',
        'bst_qh', 'mh_qh', 'zxd_qh', 'zc_qh', 'bl_qh', 'wxd_qh', 'gt_qh',
        'mg_qh', 'ms_qh', 'pvc_qh', 'zly_qh', 'de_qh', 'dp_qh', 'tks_qh',
        'jd_qh', 'lldpe_qh', 'jbx_qh', 'xwb_qh', 'jhb_qh', 'dy_qh', 'hym_qh',
        'dd_qh', 'jt_qh', 'jm_qh', 'ymdf_qh', 'ry_qh', 'lv_qh', 'xj_qh',
        'xing_qh', 'tong_qh', 'hj_qh', 'lwg_qh', 'xc_qh', 'qian_qh', 'by_qh',
        'lq_qh', 'rzjb_qh', 'xi_qh', 'ni_qh', 'qz_qh', 'gz_qh', 'sngz_qh',
        'szgz_qh', 'zzgz_qh'
    ]

    conn = pymysql.connect(
        user='root', password='', database='spec', charset='utf8')
    cursor = conn.cursor()

    cursor.execute("delete from symbols")

    qh_nameUrl = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQFuturesData?node={name}"
    i = 0

    for qh_name in qh_namesAll:
        i += 1
        if (i % 5 == 0):
            print("正在刷新symbol {i} / {c}".format(
                i=i, c=len(qh_namesAll), name=qh_name))

        response = requests.get(qh_nameUrl.format(name=qh_name))
        qh_symbolsJson = response.content.decode("gb2312")
        qh_symbols = demjson.decode(qh_symbolsJson)

        if qh_symbols is not None:
            for qh_symbol in qh_symbols:
                cursor.execute(
                    "insert into symbols (symbol, name) values (%s, %s)",
                    (qh_symbol["symbol"], qh_symbol["name"]))

    print("symbol刷新完毕")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    refresh()