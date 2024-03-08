from typing import List, Any, Tuple
import numpy as np

# def plusOne(self, digits: List[int]) -> List[int]:
    
# сложение в столбик
def string_plus(a: str, b: str)->str:
    
    toint = lambda x: ord(x)-48
    tochar = lambda x: chr(x+48)
    if len(a) > len(b):
        a,b = b,a
    a = a[::-1]
    b = b[::-1]
    c = ''
    overflow = 0    
    for i in range(len(a)):
        ci = toint(a[i]) + toint(b[i]) + overflow
        overflow = ci//10
        current = ci % 10
        c += tochar(current)
    
    for i in range(len(a),len(b)):
        ci = toint(b[i]) + overflow
        overflow = ci//10
        current = ci % 10
        c += tochar(current)
    
    if overflow > 0:
        c+='1'
    
    c = c[::-1]
    return c

def plusOne(digits: List[int]) -> List[int]:
    overflow = 1
    c = []
    for i in range(len(digits)-1, -1,-1):
        ci = digits[i]+overflow
        rem = ci % 10
        overflow = ci // 10
        c.append(rem)
    if overflow > 0:
        c.append(1)
    c.reverse()
    return c  


def test_string_plus():
    for i in range(1000):
        a = np.random.randint(low=0,high=99999999)
        b = np.random.randint(low=0,high=99999999)
        assert a + b == int(string_plus(str(a), str(b)))

def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    if len(intervals)==0:
        return [newInterval]
    left_ = -1
    right_= -1

    # find first left and last right
    if newInterval[0] < intervals[0][0]:
        # find right  
        for i in range(len(intervals)):
            if intervals[i][0] <= newInterval[1]:
                right_ = i
    elif newInterval[1] > intervals[-1][1]:
        # find left  
        for i in range(len(intervals)):
            if intervals[i][1]>=newInterval[0]:
                left_ = i
                break
    else:
        # find left and right  
        ## find right  
        # print('here')
        for i in range(len(intervals)):
            if intervals[i][0] <= newInterval[1]:
                right_ = i
        ## find left  
        for i in range(len(intervals)):
            if intervals[i][1] >= newInterval[0]:
                left_ = i
                break
    # print(left_,right_)
    
    o_ = []

    if left_ == -1 and right_ == -1:
        if newInterval[1] < intervals[0][0]:
            o_.append(newInterval)
            for i in range(len(intervals)):
                o_.append(intervals[i])
        elif newInterval[0] > intervals[-1][1]:
            for i in range(len(intervals)):
                o_.append(intervals[i])
            o_.append(newInterval)
    elif left_ == -1:
        tmp_interval_ = [newInterval[0], max(intervals[right_][1],newInterval[1])]
        o_.append(tmp_interval_)
        for i in range(right_+1,len(intervals)):
            o_.append(intervals[i])
    elif right_ == -1:
        for i in range(0,left_):
            o_.append(intervals[i])
        tmp_interval_ = [min(intervals[left_][0],newInterval[0]),newInterval[1]]
        o_.append(tmp_interval_)
    else:
        # copy left intervals
        for i in range(left_):
            o_.append(intervals[i])
        # merge and insert 
        tmp_interval_ = [min(intervals[left_][0],newInterval[0]),max(intervals[right_][1],newInterval[1])]
        o_.append(tmp_interval_)            
        # copy right intervals
        for i in range(right_+1, len(intervals)):
            o_.append(intervals[i])

    return o_



# intervals = [[1,3],[6,9]]
# intervals = [[1,5]]
# newInterval = [2,6]
# newInterval = [-10,0]
# newInterval = [8,11]
intervals = [[0,0],[2,4],[9,9]]
newInterval = [0,7]

print(insert(intervals,newInterval))




# ../python3venvs/ml/bin/python -m pytest ./main.py -v