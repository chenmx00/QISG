import tushare as ts
cons = ts.get_apis()

try:

    # bar = ts.bar('600000', conn=cons, freq='D', start_date="2015-02-08")
    print(ts.get_instrument(cons[1]))
    # print(bar)
    
finally:
    ts.close_apis(cons)