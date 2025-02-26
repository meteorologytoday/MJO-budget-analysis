import datetime


def datetime2fmon(t):
    return t.year*12 + (t.month-1)

def fmon2datetime(fmon):

    year = fmon // 12
    mon  = fmon % 12 + 1

    return datetime.datetime(year, mon, 1)


def getYearMonthFromfmon(fmon):

    dt = fmon2datetime(fmon)
    
    return dt.year, dt.month


