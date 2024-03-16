from telnetlib import theNULL
from typing import List, Any, Tuple
import numpy as np
from copy import deepcopy,copy
from pprint import pprint
import queue
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
# intervals = [[0,0],[2,4],[9,9]]
# newInterval = [0,7]
# print(insert(intervals,newInterval))

def merge(intervals: List[List[int]]) -> List[List[int]]:
    # sort intervals by left border
    # 10 9 8 7 6 
    
    # 9 8 7 6 10
    # ... 9
    # ... 8
    # ... 7

    N = len(intervals)

    for i in range(len(intervals)-1):
        for j in range(0, len(intervals)-1):
            if intervals[j][0] > intervals[j+1][0]:
                intervals[j], intervals[j+1] = intervals[j+1],intervals[j]
            
    def is_overlaps(int1 , int2):
        return (int1[0] <= int2[0] and int1[1] >= int2[0]) or (int1[0]>=int2[0] and int1[0]<=int2[1])

    def find_max_b_and_overlapped(start_, intervals):
        # suppose all sorted by left border
        if start_ >= N-1:
            return start_, start_, start_

        next_ = start_+1
        a = intervals[start_][0]
        b = intervals[start_][1]
        bmax = start_
        while(is_overlaps(intervals[start_], intervals[next_])):
            if intervals[next_][1]>b:
                bmax = next_
                b=  intervals[next_][1]
            next_ += 1
            if next_ >= N:
                break
        
        return bmax,next_-1,start_
        
    # suppose all intervals sorted by left border
    start = 0
    o = []
    while(True):
        if start > N-1:
            break 
        if start == N-1:
            o.append(intervals[start])
            break
        interval_ = [intervals[start][0]]
        biggest_connect_with_start, last_seen_connected_with_start, start = find_max_b_and_overlapped(start, intervals)
        while(True):

            biggest_connect_with_start, last_seen_connected_with_start, start = find_max_b_and_overlapped(last_seen_connected_with_start, intervals)
            if biggest_connect_with_start == start:
                # ни с чем более нет связи 
                break

        interval_.append(intervals[biggest_connect_with_start][1])
        o.append(interval_)
        start = last_seen_connected_with_start+1
        
    return o

def merge(intervals: List[List[int]]) -> List[List[int]]:
    # sort intervals by left border
    # 10 9 8 7 6 
    
    # 9 8 7 6 10
    # ... 9
    # ... 8
    # ... 7

    N = len(intervals)

    for i in range(len(intervals)-1):
        for j in range(0, len(intervals)-1):
            if intervals[j][0] > intervals[j+1][0]:
                intervals[j], intervals[j+1] = intervals[j+1],intervals[j]
            
    def is_overlaps(int1 , int2):
        return (int1[0] <= int2[0] and int1[1] >= int2[0]) or (int1[0]>=int2[0] and int1[0]<=int2[1])

    def merge_2(int1,int2):
        return [min(int1[0],int2[0]), max(int1[1],int2[1])]
    

    # suppose all intervals sorted by left border
    o = []
    pos_ = 0
    while(pos_ <= N-1):
        tmp_ = intervals[pos_]
        while(pos_ <= (N-1) and is_overlaps(tmp_, intervals[pos_])):
            tmp_ = merge_2(tmp_,intervals[pos_])
            pos_ += 1
        o.append(tmp_)
        
    return o

# intervals = [[1,3],[2,6],[8,10],[15,18]]
# testcase_1 = [[1,6],[8,10],[15,18]]
# # # intervals = [[1,4],[4,5]]
# # # testcase_1 = [[1,5]]

# # # intervals = [[1,4],[0,2],[3,5]]
# # # testcase_1 = [[0,5]]

# # # intervals = [[1,4],[2,3]]
# # # testcase_1 = [[1,4]]
# # intervals = [[5,5],[1,3],[3,5],[4,6],[1,1],[3,3],[5,6],[3,3],[2,4],[0,0]]
# # testcase_1 = [[0,0],[1,3],[4,6]]
# # intervals = [[1,4],[2,3]]
# # testcase_1=  [[1,4]]
# assert  testcase_1 == merge(intervals)


def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    # Given an array of intervals intervals where intervals[i] = [starti, endi], 
    # return the minimum number of intervals you need to remove 
    # to make the rest of the intervals non-overlapping.
    # [[1,2],[2,3],[3,4],[1,3]]
    # erase [1,3]
    intervals.sort()
    res = 0
    prevEnd = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= prevEnd:
            prevEnd = end
        else:
            res += 1
            prevEnd = min(end, prevEnd)
    
    return res


# intervals = [[5,5],[1,4],[1,3],[3,5],[4,6],[1,1],[3,3],[5,6],[3,3],[2,4],[0,0]]
# intervals_ = deepcopy(intervals)
# intervals_.sort()
# print(intervals_)
# print(eraseOverlapIntervals(intervals))
# print([1,2] < [1,3])


def maxSubArray(nums: List[int]) -> int:
    # Given an integer array nums, find the 
    # subarray
    # with the largest sum, and return its sum.
    # O(n)
    # nums.sort()
    # pos_ = -1
    # for i in range(len(nums)):
    #     if nums[i] > 0:
    #         pos_ = i
    #         break
    # if pos_ == -1:
    #     return nums[-1]
    # else:
    #     return sum(nums[pos_:]) 
    #  [-2,1,-3,4,-1,2,1,-5,4]
    #           |       |
    N = len(nums)
    left = 0
    right = 0
    csum = 0
    maxsum = nums[0]
    for i in range(N):
        csum += nums[i]
        if csum > maxsum:
            maxsum = csum 
            right = i

        if csum < 0:
            csum = 0
            left = i+1
    
    # print(left,right)
    # print(nums[left:right+1])
    # print(left,right)
    # return sum(nums[left:right+1]) 
    return maxsum

# nums = [-2,1,-3,4,-1,2,1,-5,4]
# print(nums)
# print(maxSubArray(nums))

# def try_next(currentpos, nums, stack):
#     stack.append(currentpos)
#     currentposmaxjump = nums[currentpos]
#     while(currentposmaxjump>0):
#         if (currentposmaxjump + currentpos) >= len(nums)-1:
#             stack.append(len(nums)-1)
#             return True
#         else:
#             signal = try_next(currentpos+currentposmaxjump, nums, stack)
#             if signal == True:
#                 return True
#             else:
#                 currentposmaxjump -= 1
#     stack.pop()
#     return False

# def canJump(nums: List[int]) -> bool:
#     N = len(nums)
#     if N == 0:
#         return False
#     T = N-1
#     path = []
#     result = try_next(0, nums, path)
#     print(path)
#     return result

# def try_next(currentpos, nums):
#     print(f'{(currentpos+1)/(len(nums))*100}')
#     currentposmaxjump = nums[currentpos]
#     tmp_= 1
#     while(tmp_<=currentposmaxjump):
#         if (tmp_ + currentpos) >= len(nums)-1:
#             return True
#         else:
#             signal = try_next(currentpos+tmp_, nums)
#             if signal == True:
#                 return True
#             else:
#                 tmp_ += 1
#     return False

# def canJump(nums: List[int]) -> bool:
#     N = len(nums)
#     if N <= 1:
#         return True
#     result = try_next(0, nums)
#     return result

# def canJump(nums: List[int]) -> bool:
#     N = len(nums)
#     if N <= 1:
#         return True
#     T = N-1
#     for i in range(len(nums)-1,-1,-1):
#         if nums[i]+i >= T:
#             T = i
#     return T == 0
# nums = [2,3,1,1,4]
# nums = [3,2,1,0,4]
# nums = [2,0,6,9,8,4,5,0,8,9,1,2,9,6,8,8,0,6,3,1,2,2,1,2,6,5,3,1,2,2,6,4,2,4,3,0,0,0,3,8,2,4,0,1,2,0,1,4,6,5,8,0,7,9,3,4,6,6,5,8,9,3,4,3,7,0,4,9,0,9,8,4,3,0,7,7,1,9,1,9,4,9,0,1,9,5,7,7,1,5,8,2,8,2,6,8,2,2,7,5,1,7,9,6]
# print(canJump(nums))

def subsets(nums: List[int]) -> List[List[int]]:
    combs = []
    N = len(nums)
    # basepos = 0
    # while(basepos<=N-1):
    #     tmp = []
    #     i = basepos
    #     while(len(tmp)<=(N-basepos)-1):
    #         tmp.append(nums[i])
            

    #         combs.append(deepcopy(tmp))
    #         i+=1
    #     basepos += 1

    stack = []
    def rec(pos):
        if pos==N:
            # copy stack
            combs.append(stack.copy())
            return
        stack.append(nums[pos])
        rec(pos+1)
        stack.pop()
        rec(pos+1)

    rec(0)

    return combs

# nums = [1,2,3]
# print(subsets(nums))


# numofcalls = [0]

def rec(cpos, maxstep, n, cache:dict):
    # numofcalls[0] +=1
    tmp_ = 1
    while(tmp_ <= maxstep):
        if cache[cpos+tmp_] != 0:
            cache[cpos] += cache[cpos+tmp_]
        else:
            rec(cpos+tmp_, maxstep, n,cache)
            cache[cpos] += cache[cpos+tmp_]
        tmp_ += 1


def climbStairs(n: int) -> int:
    # You are climbing a staircase. It takes n steps to reach the top.
    # Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    # i.e. count all paths
    if n == 1:
        return 1
    if n == 2:
        return 2
    cache = {k:v for k,v in zip(range(0,n),[0]*n)}
    cache[n-1] = 1
    cache[n-2] = 2
    rec(0, 2, n, cache)
    return cache[0]

# climbStairs(37)
# print(numofcalls)

def getbestpos(l,r, costs, totalcost):
    lcost = costs[l]
    rcost = costs[r]
    bestpos = None
    if lcost <= rcost:
        bestpos = l
    else:
        bestpos = r
    return bestpos, totalcost+costs[bestpos]

def rec(costs:list):
    n = len(costs)
    for i in range(n-3,-1,-1):
        costs[i] = costs[i] + min(costs[i+1], costs[i+2]) 
    return min(costs[0], costs[1])

def minCostClimbingStairs(cost: List[int]) -> int:
    return rec(cost)

# print(minCostClimbingStairs([0,1,1,0])) 
# print(minCostClimbingStairs([0,2,2,1]))
# print(minCostClimbingStairs([1,100,1,1,1,100,1,1,100,1]))
        

def rec(newcolor, i,j,c,a):
    # воду скипаем
    if a[i][j] == 0:
        return False
    # наступили на ячейку, где мы уже были
    if c[i][j] != 0:
        return False
    # еще не раскашенная земля - красим
    c[i][j] = newcolor
    # красим соседей, если они есть
    rec(newcolor, i+1,j, c,a)
    rec(newcolor, i-1,j, c,a)
    rec(newcolor, i,j+1, c,a)
    rec(newcolor, i,j-1, c,a)
    # если покрасили хотя бы одну землю в новый цвет - занчит успешно нашли остров
    return True
    

def numIslands(grid: List[List[str]]) -> int:
    n = len(grid)
    m = len(grid[0])
    a = [[0]*(m+2) for i in range(n+2)]
    c = [[0]*(m+2) for i in range(n+2)]
    
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '0':
                a[i+1][j+1] = 0
            if grid[i][j] == '1':
                a[i+1][j+1] = 1
    for i in range(n+2):
        print(a[i])
    
    ucolor = 1
    for i in range(1,n+1):
        for j in range(1,m+1):
            # раскаска в глубину
            is_painted = rec(ucolor, i,j,c,a)
            ucolor += int(is_painted)        
    return ucolor-1


# # grid = [
# #   ["1","1","1","1","0"],
# #   ["1","1","0","1","0"],
# #   ["1","1","0","0","0"],
# #   ["0","0","0","0","0"]
# # ]
# grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# print(numIslands(grid))


def rec(a,element_to_push_pos,output, buffer, cumsum, target):
    if cumsum == target:
        # find the good comb
        output.append(buffer.copy())
        pass 
    elif cumsum > target or element_to_push_pos >= len(a):
        # that combination is bad
        return 
    else:
        buffer.append(a[element_to_push_pos])
        rec(a, element_to_push_pos, output, buffer, cumsum+a[element_to_push_pos], target)
        # вышли из строчки выше 
        # либо нашли удачную комбинацию, либо провальную
        # без промежуточных вариантов
        # в любом случае нужно пробовать добавлять новый элементы a[i+1]
        # либо еще раз удалять текущий
        # теперь работаем только с новыми элементами, которых не видели раньше
        # но со старым базовым массивом

        # подчищаем последнюю удачную комбинацию
        buffer.pop()
        # пробуем новое
        rec(a, element_to_push_pos+1, output, buffer, cumsum,target)





def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    # найти все подмножества, которые суммируются в таргет
    a = candidates
    combs = []
    rec(a,0,combs,[], 0,target)
    # for c in combs:
    #     print(c)
    return combs
# candidates = [2,3,6,7]
# target = 7

# combinationSum(candidates, target)


# baseset is set of indicies from which we can get element from any position
def rec(a, baseset, output, buffer):
    # проверка происходит перед тем, как опустошится массив индексов
    # просто чтобы сэкономить число вызовов функций
    if len(baseset) == 1:
        buffer.append(a[baseset[0]])
        output.append(buffer.copy())
        buffer.pop()
        return

    for index in baseset:
        buffer.append(a[index])
        BminusI = baseset.copy()
        BminusI.remove(index)
        rec(a, BminusI, output, buffer)
        buffer.pop()

def permute(nums: List[int]) -> List[List[int]]:
    # return all possible permutations
    output = []
    rec(nums, [i for i in range(len(nums))], output, [])
    return output

# nums = [1,2,3]
# o_ = permute(nums)
# for el in o_:
#     print(el)

def lengthOfLongestSubstring(s: str) -> int:
    # Given a string s, find the length of the longest 
    # substring without repeating characters.
    currentset = dict()
    max_length = 0
    currentlength = 0
    q = []
    for i in range(len(s)):
        if s[i] in currentset:
            # hoops, here a repeating symbol
            currentlength -= currentset[s[i]]
            while True:
                if q[0] == s[i]:
                    del currentset[q[0]]
                    q.pop(0)
                    break
                else:
                    del currentset[q[0]]
                    q.pop(0)
            currentlength+=1
            currentset.update({s[i]:currentlength})
            q.append(s[i])
        elif not s[i] in currentset:
            currentlength+=1
            currentset.update({s[i]:currentlength})
            q.append(s[i])

        max_length = max(max_length, len(currentset))

    return max_length

# # s = "abcabcbb"
# s= "pwwkew"
# # s = "dvdf"
# assert lengthOfLongestSubstring(s) == 3

# s = 'aab'
# print(lengthOfLongestSubstring(s))
# assert lengthOfLongestSubstring(s) == 2












# ../python3venvs/ml/bin/python -m pytest ./main.py -v
# /home/user/work/penv/bin/python -m pytest ./main.py -v 