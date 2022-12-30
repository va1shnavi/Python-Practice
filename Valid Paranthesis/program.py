'''
    ****USE STACK****
    
    Summary:-
        Add opening brakets to stack. Pop when closing bracket encountered. Check if popped and
        closing brackets match. If not then return false. Check if stack is empty at end, if
        not then return false.
'''

class Solution:
    def isValid(self, s: str) -> bool:
        
        s = []
        self.stack = []

        for i in s:
            for j in s:
                if i == '(' and j == ')':
                    self.stack.append(s)
                else:
                    print('wrong input')

                if i == '[' and j == ']':
                    self.stack.append(s)

                if i == '{' and j == '}':
                    self.stack.append(s)

        try:
            self.stack.isempty()
            print('stack empty')
        except:
            print('stack has args')

        
