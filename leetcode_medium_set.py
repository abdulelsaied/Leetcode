from collections import defaultdict, deque

"""
Notes:
- build-in python sort has a runtime of O(NlogN), and O(logn) space due to recursive calls by QuickSort
- ord(char) - ord('a') gives the number of letters after "a" a letter is, in lowercase.
- tuple("hey") = ("h", "e", "y")
- O(n^2) substrings of a string of length n 
- BT Inorder Traversal - left, root, right
- BT Preorder Traversal - root, left, right
- BT Postorder Traversal - left, right, root
- "".join(str_array) converts an array of characters into a string

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

###########################################################################################################################
###########################################################################################################################

def longest_palindrome(s):
    """ Given a string s, return the longest palindromic substring in s.

    Approach 1: (brute force)
    - Choose all possible start/end pairs, and check if they are palindromes
    - Runtime: O(n^3)
    - Space: O(1)

    Approach 2: 
    - Find the longest common substring between S and S', where S' is the reverse of S
    - Check if the indices of both substrings are the same to avoid the case where the palindrome is invalid
    - Runtime: O(n^2)
    - Space: O(n^2)

    Approach 3 (⭐): 
    - Expand around all the 2n - 1 possible centers of the string
    - For each center, validate that the center is a palindrome (trivial if center is 1 char, 1 comparison if its 2 chars)
    - Expand the left and right by only comparing the outermost chars
    - Maintain max start/end indices used to slice the string in the end
    - Runtime; O(n^2)
    - Space: O(1)
    """
    if len(s) < 1:
        return ""
    start = 0 
    end = 0 
    for i in range(len(s)):
        len1 = expandCenter(s, i, i)
        len2 = expandCenter(s, i, i + 1)
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    return s[start : end + 1]
        
def expandCenter(s, left, right):
    while (left >= 0 and right < len(s) and s[left] == s[right]):
        left -= 1
        right += 1
    return right - left - 1

###########################################################################################################################
###########################################################################################################################

def increasing_triplet(nums):
    """ Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k].

    Approach 1 (⭐):
    - Keep track of instances where you see the first and second numbers of a sequence. 
    - If you ever come across a number larger than the previous first and second numbers, there must be an increasing triplet
    - The value of first when returning True may be inaccurate; maintain a placeholder for prev_first if you need to return triplet
    - Runtime: O(n)
    - Space: O(1)
    """
    first = float("inf")
    second = float("inf")
    for n in nums:
        if n <= first:
            first = n 
        elif n <= second: 
            second = n 
        else:
            return True
    return False
        
###########################################################################################################################
###########################################################################################################################

def find_missing_ranges(nums, lower, upper):
    """ Return the smallest sorted list of ranges that cover every missing number exactly. That is, no element of nums is in any of the ranges, and each missing number is in one of the ranges.

    - Approach 1 (⭐):
    - Do a linear scan of nums, appening lower - 1 and upper + 1 to the ends
    - Format depends on value of difference between consecutive elements
    - Runtime: O(n)
    - Space: O(1)
    """
    res = []
    new_arr = [lower - 1] + nums + [upper + 1]
    for i in range(len(new_arr) - 1):
        diff = new_arr[i + 1] - new_arr[i]
        if diff == 2:
            res.append(str(new_arr[i] + 1))
        elif diff > 2: 
            res.append(str(new_arr[i] + 1) + "->" + str(new_arr[i + 1] - 1))
    return res
    
###########################################################################################################################
###########################################################################################################################

def add_two_numbers(l1, l2):
    """ The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

    - Approach 1: 
    - Maintain a carry variable and perform addition from left to right, adding values to a new LL
    - Runtime: O(n)
    - Space: O(1)

    - Approach 2 (⭐):
    - Use the longer LL as the result
    - First, find the longest LL (O(n))
    - Perform addition from left to right. 
    - Make sure to account for the case where LL end is reached and carry == 1
    - Runtime: O(n)
    - Space: O(1)
    """
    first_len = 0
    second_len = 0 
    first_ptr = l1
    second_ptr = l2
    while first_ptr != None:
        first_len += 1 
        first_ptr = first_ptr.next
    while second_ptr != None:
        second_len += 1
        second_ptr = second_ptr.next
    if first_len >= second_len:
        max_ptr = l1
        min_ptr = l2
    else: 
        max_ptr = l2
        min_ptr = l1
    carry = 0 
    while max_ptr != None:
        val = max_ptr.val + min_ptr.val if min_ptr else max_ptr.val
        if val + carry > 9:
            max_ptr.val = (val + carry) % 10
            carry = 1
        else:
            max_ptr.val = val + carry
            carry = 0 
        if max_ptr.next == None and carry == 1:
            max_ptr.next = ListNode(1, None)
            carry = 0
        max_ptr = max_ptr.next
        if min_ptr:
            min_ptr = min_ptr.next
    return l1 if first_len >= second_len else l2

###########################################################################################################################
###########################################################################################################################

def odd_even_list(head):
    """ Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

    - Approach 1 (⭐):
    - Put all odd nodes in one LL and the even in another, then link them in the end
    - To do this in-place, rewire the nodes in pairs rather than using new LL's 
    - Each linked list should have a head and tail pointer, with tail pointer acting as the iterator
    - Runtime: O(n)
    - Space: O(1)
    """
    if head is None:
        return head
    even = head.next
    evenHead = even
    odd = head
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = evenHead
    return head

###########################################################################################################################
###########################################################################################################################

def get_intersection_node(headA, headB):
    """ Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. 

    - Approach 1 (brute force):
    - For each node in headA, check all nodes in headB for a match
    - Runtime: O(M x N)
    - Space: O(1)

    - Approach 2:
    - Traverse B and store the nodes in a hash table. 
    - Iterate over A and check for matches in the table
    - Runtime: O(M + N)
    - Space: O(max(M, n)) (depends on which LL we choose to hash)

    - Approach 3 (⭐): 
    - Find the lengths of both LL's --> O(m + n)
    - Set a pointer to start at the first possible point an intersection can occur based on the length of the shorter LL
    - Now, iterate over both LL's and pairwise-compare.
    - EXTRA: traverse both LL's at the same time, jumping to the next LL when first one is traversed
    - a + c + b = b + c + a
    - By the time they meet, it will either be at the point of intersection or when both are null, since the length of both paths are the same
    - Runtime: O(M + M)
    - Space: O(1)
    """
    a_ptr = headA
    b_ptr = headB
    while a_ptr != b_ptr:
        a_ptr = headB if not a_ptr else a_ptr.next
        b_ptr = headA if not b_ptr else b_ptr.next
    return a_ptr

###########################################################################################################################
###########################################################################################################################

def inorder_raversal(root):
    """ Given the root of a binary tree, return the inorder traversal of its nodes' values.

    - Inorder Traversal: left, root, right

    - Approach 1 (recursive):
    - Base case: root is null -> return []
    - [left] + [root] + [right]
    - Runtime: O(n)
    - Space: O(n)

    - Approach 2 (⭐):
    - To perform a recursive task iteratively, utilize a stack
    - First add all left nodes on far left. 
    - Continously pop off stack, going right at the end and searching far left if possible
    - Runtime: O(n)
    - Space: O(n)
    """
    res = []
    stack = []
    curr = root
    while curr or len(stack) > 0:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res

###########################################################################################################################
###########################################################################################################################

def zigzag_level_order(root):
    """ Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

    - Approach 1 (⭐) BFS:
    - Maintain a deque and add nodes in level-order fashion. 
    - Maintain a second deque that contains the nodes for 1 level in the order needed for that specific level
    - Runtime: O(n)
    - Space: O(n)
    """
    ret = []
    level_list = deque()
    if not root:
        return ret
    node_queue = deque([root, None])
    is_order_left = True
    while node_queue:
        curr_node = node_queue.popleft()
        if curr_node:
            if is_order_left:
                level_list.append(curr_node.val)
            else:
                level_list.appendleft(curr_node.val)
            if curr_node.left:
                node_queue.append(curr_node.left)
            if curr_node.right:
                node_queue.append(curr_node.right)
        else:
            ret.append(level_list)
            if node_queue:   
                node_queue.append(None)
            level_list = deque()
            is_order_left = not is_order_left
    return ret

###########################################################################################################################
###########################################################################################################################

def build_tree(preorder, inorder):
    """ Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

    - Approach 1 (⭐):
    - Notice that the preorder array always contains the root of the tree as it's first element
    - Notice that the inorder array always contains left/right subtree on the left/right side of root found in preorder. 
    - So, you can construct the tree recursively by using the first element of preorder, finding it in inorder, and building the tree from there. 
    - Can map values in inorder so that searching for preorder[0] in inorder is O(1)
    - Runtime: O(n)
    - Space: O(n)
    """
    preorder_index = 0
    inorder_index_map = {}
    for index, value in enumerate(inorder):
        inorder_index_map[value] = index
    def array_to_tree(left, right):
        nonlocal preorder_index
        if left > right: 
            return None
        root_value = preorder[preorder_index]
        root = TreeNode(root_value)
        preorder_index += 1
        root.left = array_to_tree(left, inorder_index_map[root_value] - 1)
        root.right = array_to_tree(inorder_index_map[root_value] + 1, right)
        return root
    return array_to_tree(0, len(preorder) - 1)

###########################################################################################################################
###########################################################################################################################

def connect(root):
    """ You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. Populate each next pointer to point to its next right node.

    - Approach 1: 
    - Traverse the tree in level order, maintaining a queue that stores the nodes of each level. 
    - When we pop a node from the queue, set it's next to the next item in the queue if its not at the end of the level
    - Runtime: O(n)
    - Space: O(n)

    - Approach 2 (⭐):
    - Notice that there are two types of connections: left pointing to its right sibling, or right pointing to the left of its parent's sibling. 
    - Each case can be handled one level in advance
    - Runtime: O(n)
    - Space: O(1)
    """
    if not root:
        return None
    leftmost = root
    while leftmost.left:
        head = leftmost
        while head:
            head.left.next = head.right
            if head.next:
                head.right.next = head.next.left
            head = head.next
        leftmost = leftmost.left
    return root

###########################################################################################################################
###########################################################################################################################

def kth_smallest(root, k):
    """ Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

    - Approach 1 (⭐):
    - For a BST, the inorder traversal returns the nodes in ascending order. 
    - So, do the inorder traversal, stopping at the kth iteration. 
    - Runtime: O(H + k) - H is the height
    - Space: O(H)
    """
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if not k:
            return root.val
        root = root.right

###########################################################################################################################
###########################################################################################################################

def inorder_successor(root, p):
    """ Given the root of a binary search tree and a node p in it, return the in-order successor of that node in the BST.

    - Approach 1: 
    - Using node p, there are two possibilites for where the inorder successor could be. 
    - If p has a right child, the successor is the leftmost node rooted at the right child
    - If p doesn't have a right child, run inorder traversal as normal and return successor
    - Runtime: O(n)
    - Space: O(n)

    - Approach 2 (⭐):
    - Using BST properties, we can decide which subtree the successor is from a comparison with the root. 
    - If p's val is less than the root, the successor is either the root or in the left subtree
    - If p's val is greater or equal to the root, the successor is in the subtree, but can't be the root. 
    - Runtime: O(N) if not balanced, O(logN) if balanced.
    - Space: O(1)
    """
    successor = None
    while root:
        if p.val >= root.val:
            root = root.right
        else:
            successor = root
            root = root.left
    return successor

###########################################################################################################################
###########################################################################################################################

def num_islands(grid):
    """ Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

    - Approach 1 BFS (⭐):
    - Iterate over all elements in the grid
    - If you encounter a "1", run BFS on it by adding to a queue, and adding all of its neighbors to the queue until the queue is empty
    - When we pop a node from the queue, set it to 0
    - Runtime: O(M x N)
    - Space: O(min(M, N)

    - Approach 2 DFS:
    - Iterate over all elements in the grid
    - If you encounter a "1", run DFS on it and mark all of its neighbors as "0", or set visited array if you can't modify input
    - Return count
    - Runtime: O(M x N)
    - Space: O(M x N)
    """
    count = 0 
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                count += 1
                print(i, j)
                visit_island(grid, i, j)
    return count
    
def visit_island(grid, i, j):
    grid[i][j] = "0"
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0))
    for di, dj in offsets:
        new_i = i + di
        new_j = j + dj
        if 0 <= new_i < len(grid) and 0 <= new_j < len(grid[0]) and grid[new_i][new_j] == "1":
            visit_island(grid, new_i, new_j)

###########################################################################################################################
###########################################################################################################################

def letter_combinations(digits):
    """ Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

    - Approach 1 (⭐):
    - Follows the standard backtracking template
    - Starting with the first letter of digits, add all possible numbers of that digit to path, backtrack, then pop from path
    - Runtime: O(4^N * N)
    - Space: O(N) - at most N frames on the recursive stack at a time.
    """
    if not digits: 
        return []
    letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    def backtrack(index, path):
        if len(path) == len(digits):
            combinations.append("".join(path))
            return
        possible_letters = letters[digits[index]]
        for letter in possible_letters:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()
    combinations = []
    backtrack(0, [])
    return combinations

###########################################################################################################################
###########################################################################################################################

def generate_parenthesis(n):
    """ Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

    - Approach 1:
    - Brute force; find all the 2^2n sequences of parentheses, then validate them.
    - Runtime: O((2^2n)*n)
    - Space: O((2^2n)*n)

    - Approach 2 (⭐):
    - Use backtracking approach
    - Maintain a count of left and right paren between recursive calls
    - If there are less than n left paren, we can add one, backtrack, and pop
    - If there are less right paren than left, we can add one, backtrack, and pop
    - Once the sequence is 6 chars, we can append it to the result and return 
    - Runtime: O(4^n/sqrt(n))
    - Space: O(4^n/sqrt(n))
    """
    ans = []
    def backtrack(sequence, left, right):
        if len(sequence) == 2 * n:
            ans.append("".join(sequence))
            return 
        if left < n:
            sequence.append("(")
            backtrack(sequence, left + 1, right)
            sequence.pop()
        if right < left:
            sequence.append(")")
            backtrack(sequence, left, right + 1)
            sequence.pop()  
    backtrack([], 0, 0)
    return ans

###########################################################################################################################
###########################################################################################################################

def permute(nums):
    """ Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

    - Approach 1 (⭐):
    - Use backtracking technique; for each letter, find all permutation for the rest of the list, and add the letter to each permutations. 
    - Then, add the letter back to the list and choose the next letter to be the letter removed. 
    - Runtime: <= O(N x N!) - for each k, perform N(N - 1)(N - 2)...(N - k + 1) operations
    - Space: O(N!) - storing N! solutions   
    """
    result = []
    if len(nums) == 1:
        return [nums[:]]
    for i in range(len(nums)):
        x = nums.pop(0)
        perms = permute(nums)
        for perm in perms:
            perm += [x]
        result += perms
        nums.append(x)
    return result

###########################################################################################################################
###########################################################################################################################

def subsets(nums):
    """ Given an integer array nums of unique elements, return all possible subsets (the power set).

    Approach 1 (⭐):
    - Use the backtracking approach over all possible lengths, from 0 to n 
    - Runtime: O(N x 2^N)
    - Space: O(N)
    """
    def backtrack(start, curr):
        ans.append(curr[:])
        for i in range(start, n):
            curr.append(nums[i])
            backtrack(i + 1, curr)
            curr.pop()
    n = len(nums)
    ans = []
    backtrack(0, [])
    return ans

###########################################################################################################################
###########################################################################################################################

def exist(self, board, word):
    """ Given an m x n grid of characters board and a string word, return true if word exists in the grid.

    - Approach 1 (⭐):
    - Run DFS on the graph, with all elements being possible starting points.
    - In each backtrack call, check if the current element is in bounds and equals the next letter in suffix
    - If so, then we can clear that element, dfs on the next layer, then restore the element (backtracking!)
    - If any of those calls are True, we found a path
    - Runtime: O(M x N x 3^L), where L is the length of the word to be matched
    - Space: O(L) - the maximum length of the recursive stack is L 
    """
    self.board = board
    for i in range(len(self.board)):
        for j in range(len(self.board[0])):
            if self.backtrack(i, j, word):
                return True
    return False

def backtrack(self, i, j, suffix):
    if len(suffix) == 0:
        return True
    if not (0 <= i < len(self.board)) or not (0 <= j < len(self.board[0])) or suffix[0] != self.board[i][j]:
        return False
    ret = False
    self.board[i][j] = "#"
    for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        ret = self.backtrack(i + di, j + dj, suffix[1:])
        if ret: 
            break
    self.board[i][j] = suffix[0]
    return ret

###########################################################################################################################
###########################################################################################################################

def sort_colors(nums):
    """ Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
    
    - Approach 1 (⭐):
    - Dutch National Flag Problem
    - Maintain three pointers: right edge of 0's, curr element, and left edge of 2's
    - If element is a 0 or 2, swap with curr
    - If its a 1, just move curr forward
    - Return when curr reaches the boundary of 2's
    - Runtime: O(n)
    - Space: O(1)
    """
    left = 0 
    right = len(nums) - 1
    curr = 0 
    while curr <= right:
        if nums[curr] == 0:
            nums[left], nums[curr] = nums[curr], nums[left]
            left += 1
            curr += 1
        elif nums[curr] == 2:
            nums[right], nums[curr] = nums[curr], nums[right]
            right -= 1
        else:
            curr += 1
        