class Solution:
    def romanToInt(self, s: str) -> int:

        #defining a dictionary to fetch values
        roman_to_int = {
        'I' : 1,
        'V' : 5,
        'X' : 10,
        'L' : 50,
        'C' : 100,
        'D' : 500,
        'M' : 1000 }

        #initializing a for loop and replacing the input to values from the dictionary
        for_sum = 0
        s.replace('IV', 'IIII').replace('IX','VIIII'). replace('XL','XXXX').replace('XC','LXXXX').replace('CD','CCCC').replace('CM','DCCCC')

        for char in s:
            for_sum += roman_to_int[char]
        return for_sum

