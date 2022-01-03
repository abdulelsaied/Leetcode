from collections import defaultdict

"""
Notes:
- build-in python sort has a runtime of O(NlogN), and O(logn) space due to recursive calls by QuickSort
- ord(char) - ord('a') gives the number of letters after "a" a letter is, in lowercase.
- tuple("hey") = ("h", "e", "y")
- O(n^2) substrings of a string of length n 

"""

def three_sum(nums):
    """ Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

    Approach 1 (⭐):
    - Sort the array 
    - For each element nums[i], iterate over the rest of the array and find the 2Sums, using the two pointer approach. 
    - If nums[i] > 0, it can't be the start of a new triple, and check for duplicates since they are adjacent
    - For two pointer 2Sum, iterate left after finding a triple to pass over duplicates
    - Runtime: O(n^2) --> O(n) for 2Sum done for n elements
    - Space: O(logn)/O(n) --> space required for Quicksort

    Approach 2: 
    - Sort the array 
    - For each element nums[i], iterate over the rest of the array and find the 2Sums, using the hash approach.
    - If nums[i] > 0, it can't be the start of a new triple, and check for duplicates since they are adjacent
    - For hash 2Sum, pass in the rest of the array (nums[i + 1:]) and find two numbers that sum to  -nums[i]
    - Runtime: O(n^2) --> same as approach 1
    - Space: O(n) --> space used by hashset

    Approach 3:
    - Do not sort the array
    - Use a hashset to skip over duplicates in the outer loop.
    - Iterate over the rest of the list, checking if the complement is in a hashmap, and the value corresponds to the current index
    - This allows you to reuse one hashmap.
    - Runtime: O(n^2) --> same as approach 1
    - Space: O(n) --> space used by hashset/hashmap
    """
    result = []
    length = len(nums)
    if length < 3:
        return result
    nums.sort() 
    for i in range(length):
        element = nums[i]
        if element > 0:
            break
        if i == 0 or nums[i - 1] != element:
            two_sum(nums, i, result)
    return result
                
def two_sum(nums, i, result):
    left, right = i + 1, len(nums) - 1
    while left < right:
        current_sum = nums[i] + nums[left] + nums[right]
        if current_sum < 0:
            left += 1
        elif current_sum > 0:
            right -= 1
        else:
            result.append([nums[i], nums[left], nums[right]])
            left += 1
            right -= 1
            while left < right and nums[left] == nums[left - 1]:
                left += 1

###########################################################################################################################
###########################################################################################################################

def set_zeroes(matrix) -> None:
    """ Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's, and return the matrix.

    Do not return anything, modify matrix in-place instead.

    Approach 1:
    - Loop through the entire matrix, noting the row and column of every element that is 0 in 2 sets.
    - On the next iteration of the matrix, if the element shares a row or column with the previously stored r/c, set to 0
    - Runtime: O(M x N) --> iterating through the entire M x N matrix
    - Space: O(M + N) --> space for the row and col sets

    Approach 2 (⭐):
    - Use the first cell of each row/col as a flag for the entire row/col
    - In the case of the first col, if there was a 0 contained in matrix[i][0], then that column should be 0'ed out (is_col = True).
    - In the case of the first row, if there was a 0 contained in matrix[0][j], then that row should be 0'ed out (matrix[0][0] = 0).
    - Runtime: O(M x N) --> iterating through the entire M x N matrix
    - Space: O(1) --> no extra space used, except 1 variable is_col
    """
    is_col = False
    row_length = len(matrix)
    col_length = len(matrix[0])
    for i in range(row_length):
        if matrix[i][0] == 0:
            is_col = True
        for j in range(1, col_length):
            if matrix[i][j] == 0:
                matrix[0][j] = 0 # flag for which columns should be 0'ed out 
                matrix[i][0] = 0 # flag for which rows should be 0'ed out         
                
    for i in range(1, row_length):
        for j in range(1, col_length):
            if not matrix[i][0] or not matrix[0][j]:
                matrix[i][j] = 0

    if matrix[0][0] == 0:
        for j in range(col_length):
            matrix[0][j] = 0
            
    if is_col:
        for i in range(row_length):
            matrix[i][0] = 0

###########################################################################################################################
###########################################################################################################################

def group_anagrams(strs):
    """ Given an array of strings strs, group the anagrams together. You can return the answer in any order.

    Approach 1: 
    - Sort each of the strings, and use that as a key in a hashtable. 
    - Return values of hashtable
    - Runtime: O(NKlogK), where K is the length of the longest string
    - Space: O(NK)

    Approach 2: 
    - Keep a count of each letter instance for each string, and use that as a key in a hashtable
    - Return values of hashtable
    - Runtime: O(NK), since we aren't sorting the strings anymore
    - Space: O(NK)
    """
    ans = defaultdict(list)
    for word in strs:
        counts = [0] * 26
        for char in word:
            counts[ord(char) - ord('a')] += 1
        ans[tuple(counts)].append(word)
    return ans.values()

###########################################################################################################################
###########################################################################################################################

def longest_substring_no_duplicates(s):
    """ Given a string s, find the length of the longest substring without repeating characters.

    Approach 1: (brute force)
    - For all O(n^2) substrings, check if there are no matching characters
    - Runtime: O(n^3) 
    - Space: O(min(m, n)) where m, n are the size of the string and character set

    Approach 2: 
    - Sliding window technique with 2 pointers
    - Maintain a set initialized to [0] * len(charset)
    - If the next letter is already in the set, move left pointer
    - Runtime: O(2n) = O(n)
    - Space: O(min(m, n)) - size of the set is upper bounded by size of string and charset

    Approach 3 (⭐):
    - Similar to sliding window, but maintain a map of chars to indices 
    - Instead of moving left pointer one-by-one, jump it straight to the index right after the index listed in the map
    - Runtime: O(n)
    - Space: O(min(n, m))
    """
    chars = [0] * 128
    i = 0
    j = 0 
    res = 0 
    while i < len(s):
        chars[ord(s[i])] += 1 # add s[i] to the set
        while chars[ord(s[i])] > 1: # if s[i] was already in the set -> duplicate
            chars[ord(s[j])] -= 1
            j += 1
        res = max(res, i - j + 1)
        i += 1
    return res



