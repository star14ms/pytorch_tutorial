import time as t
import datetime as dt
from selenium import webdriver


class Time:

    def now():
        return t.perf_counter()

    def sec_to_hms(second):
        second = int(second)
        h, m, s = (second // 3600), (second//60 - second//3600*60), (second % 60)
        return h, m, s
    
    def hms_delta(start_time, hms=False, rjust=False, join=':'):
        time_delta = t.perf_counter() - start_time
        h, m, s = Time.sec_to_hms(time_delta)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")

    def hms(second, hms=False, rjust=False, join=':'):
        h, m, s = Time.sec_to_hms(second)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")


def alerm(webdriver_path=r'C:\Users\danal\Documents\programing\chromedriver.exe', loading_sec=7):
    now = dt.datetime.today()

    if int(now.strftime('%S')) < 60 - loading_sec:
        alarm_time = now + dt.timedelta(minutes=1)
    else:
        alarm_time = now + dt.timedelta(minutes=2)

    alarm_time = alarm_time.strftime('%X')
    driver = webdriver.Chrome(webdriver_path)
    driver.get(f'https://vclock.kr/#time={alarm_time}&title=%EC%95%8C%EB%9E%8C&sound=musicbox&loop=1')
    driver.find_element_by_xpath('//*[@id="pnl-main"]').click()
    input('\033[1mPress Enter\033[0m')