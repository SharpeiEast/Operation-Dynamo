# 寻找混乱数组中的第K小（大的值）
# O(n) time, quick selection
# 这个k是从零开始数的，可以理解为这堆数如果排好序，那么sorted_nums[k]是谁。
def findKthLargest(nums, k):
    # convert the kth largest to smallest
    start = time.time()
    rst = findKthSmallest(nums, len(nums)+1-k)
    t = time.time() - start
    return rst, len(nums), t
    
def findKthSmallest(nums, k):
    if len(nums) < k:
        return -1
    if nums:
        pos = partition(nums, 0, len(nums)-1)
        if k > pos+1:
            return findKthSmallest(nums[pos+1:], k-pos-1)
        elif k < pos+1:
            return findKthSmallest(nums[:pos], k)
        else:
            return nums[pos]
 
# choose the right-most element as pivot   
# r的索引对应的数被用来做比较的，l每次都会前移，但是low只有nums[l]大于nums[r]
# 的时候才会前移，所以最后low之前的数字都是大于nums[r]的，l会走完整个数列到达r
# 的位置然后跳出整个循环，这时把nums[r]替换到nums[low]的位置就行了，前面的数
# 都比它小，后面的数都比它大。这个函数虽然没有return nums，但是其实对它进行了操作
def partition(nums, l, r):
    low = l
    while l < r:
        if nums[l] < nums[r]:
            nums[l], nums[low] = nums[low], nums[l]
            low += 1
        l += 1
    nums[low], nums[r] = nums[r], nums[low]
    return low

# print(findKthSmallest([0,3,2,1,4,7,6,5], 6))
# 数硬币的收集次数
def minSteps(height):
    
    def minStepHelper(height, left, right, h):
        if left >= right:
            return 0
        
        m = left
        for i in range(left, right):
            if height[i] < height[m]:
                m = i
         
        return min(right - left, 
                   minStepHelper(height, left, m, height[m]) +
                   minStepHelper(height, m + 1, right, height[m]) +
                   height[m] - h)
    
    return minStepHelper(height, 0, len(height), 0)   

# height = [3, 1, 2, 5, 1]
# minSteps(height)

# 两个排好序的数组拼接之后的中位数
# 类归并排序，寻找一个m1和m2点，使得nums[m1]和nums[m2]正好作为分界点，将两个list
# 分成左右两部分，需要满足条件是nums1[m1]小于nums2[m2+1]，nums2[m2]小于nums1[m1+1]
# m1 + m2 = (n1 + n2 + 1) // 2
class Solution:
    def findMedianSortedArrays(self, nums1, nums2) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        if n1 > n2:
            return self.findMedianSortedArrays(nums2,nums1)
        k = (n1 + n2 + 1)//2
        left = 0
        right = n1
        while left < right :
            m1 = left +(right - left)//2
            m2 = k - m1
            if nums1[m1] < nums2[m2-1]:
                left = m1 + 1
            else:
                right = m1
        m1 = left
        m2 = k - m1 
        c1 = max(nums1[m1-1] if m1 > 0 else float("-inf"), nums2[m2-1] if m2 > 0 else float("-inf") )
        if (n1 + n2) % 2 == 1:
            return c1
        c2 = min(nums1[m1] if m1 < n1 else float("inf"), nums2[m2] if m2 <n2 else float("inf"))
        return (c1 + c2) / 2
# s = Solution()
# s.findMedianSortedArrays([1,2,3,4,5],[6,7,8,9])

# Count of Smaller Numbers after self
def countSmaller(nums):
    def sort(enum):
        half = len(enum) // 2
        if half:
            left, right = sort(enum[:half]), sort(enum[half:])
            m, n = len(left), len(right)
            i = j = 0
            while i < m or j < n:
                if j == n or i < m and left[i][1] <= right[j][1]:
                    smaller[left[i][0]] += j
                    enum[i+j] = left[i]
                    i += 1
                else:
                    enum[i+j] = right[j]
                    j += 1
            print("left: ", left)
            print("right: ", right)
            print("smaller: ", smaller)
        print("enum: ", enum)
        return enum
    smaller = [0] * len(nums)
    sort(list(enumerate(nums)))
    return smaller

# nums = [5, 2, 6, 1]
# countSmaller(nums)



# 计算逆序数1
def merge_sort(a):
    
    s = 0
    
    if len(a) <= 1: return 0
    
    mid = len(a) // 2
    
    l = a[:mid]
    r = a[mid:]
    
    s += merge_sort(l) + merge_sort(r)
    
    i = j = k = 0
    
    while(i < len(l) and j < len(r)):
        if(l[i] <= r[j]):
            a[k] = l[i]
            i += 1
            k += 1
        else:
            a[k] = r[j]
            j += 1
            k += 1
            s += len(l) - i     
    while(i < len(l)):
        a[k] = l[i]
        i += 1
        k += 1
    while(j < len(r)):
        a[k] = r[j]
        j += 1
        k += 1
    
    return s

# n = int(input())
# a = [int(i) for i in input().split()]

# print(merge_sort([1,2,3,4,5,6,7,0,0]))

# 逆序数，万门课程方法
def merge(left,right):
    result = list()
    i,j = 0,0
    inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        elif right[j] < left[i]:
            result.append(right[j])
            j += 1
            inv_count += (len(left)-i)
    result += left[i:]
    result += right[j:]
    return result,inv_count

# O(nlgn)
def countInvFast(array):
    if len(array) < 2:
        return array, 0
    middle = len(array) // 2
    left,inv_left = countInvFast(array[:middle])
    right,inv_right = countInvFast(array[middle:])
    merged, count = merge(left,right)
    count += (inv_left + inv_right)
    return merged, count


# 合并区间autox
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
    
    def __str__(self):
        return "[" + self.start + "," + self.end + "]"
    
    def __repr__(self):
        return "[%s, %s]" % (self.start, self.end)
def merge_interval(intervals):
    intervals.sort(key=lambda x: x.start)

    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1].end < interval.start:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1].end = max(merged[-1].end, interval.end)
    return merged
i1 = Interval(1,9)
i2 = Interval(2,5)
i3 = Interval(19,20)
i4 = Interval(10,11)
i5 = Interval(12,20)
i6 = Interval(0,3)
i7 = Interval(0,1)
i8 = Interval(0,2)
# intervals = [i1,i2,i3,i4,i5,i6,i7,i8]
# print(merge_interval(intervals))

# intervals.sort(key=lambda x: x.start)
# print(intervals)

# 摔鸡蛋问题进阶解法
class Solution:
    def superEggDrop(self, K, N):
        # K egg, N floor;
        dp = [[0 for _ in range(N + 1)] for _ in range(K + 1)]
        for i in range(1, K + 1):
            for step in range(1, N + 1):
                dp[i][step] = dp[i - 1][step - 1] + (dp[i][step - 1] + 1)
                if dp[K][step] >= N:
                    return step
        return 0
# SS = Solution()
# print(SS.superEggDrop(2,5000))
# 摔鸡蛋问题朴素解法
def superEggDrop(K: int, N: int):

    memo = dict()
    def dp(K, N) -> int:
        # base case
        if K == 1: return N
        if N == 0: return 0
        # 避免重复计算
        if (K, N) in memo:
            return memo[(K, N)]

        res = float('INF')
        # 穷举所有可能的选择
        for i in range(1, N + 1):
            res = min(res, 
                      max(
                            dp(K, N - i), 
                            dp(K - 1, i - 1)
                         ) + 1
                  )
        # 记入备忘录
        memo[(K, N)] = res
        return res

    return dp(K, N)
# print(superEggDrop(3, 100))

