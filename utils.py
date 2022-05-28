import time as t


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