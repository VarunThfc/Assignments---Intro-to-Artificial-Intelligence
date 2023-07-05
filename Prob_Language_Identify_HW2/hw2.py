import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
#     #Using a dictionary here. You may change this to any data structure of
#     #your choice such as lists (X=[]) etc. for the assignment
#     X=dict()
#     with open (filename,encoding='utf-8') as f:
#         # TODO: add your code here

#     return X
    freq = {}
    with open(filename,encoding='utf-8') as f:
        for line in f:
            for ch in line:
                ch = ch.upper()
                if(ch >= chr(65) and ch <= chr(90)):
                    if(freq.get(ch) == None):
                        freq[ch] = 1
                    else:
                        freq[ch] = freq[ch] + 1;
        f.close()
    print('Q1')
    for i in range(0,26):
        num = freq.get(chr(i+65)) if freq.get(chr(i+65)) != None else 0 
        print(chr(i+65), num)
    
    return freq;

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

#1.1
freq = shred('samples/letter.txt')
#1.2
#Priors
prob_english = 0.6
prob_spanish = 0.4
e, s = get_parameter_vectors()

prob_eng_given_x = 0
prob_sp_given_x = 0
prog_eng_given_x_a = 0
prob_sp_given_x_a = 0

for i in range(0,26):
    num = freq.get(chr(i+65)) if freq.get(chr(i+65)) != None else 0 
    if(i == 0):
        #1.2
        print("Q2")
        print(f"{num * math.log(e[i]):.4f}")
        print(f"{num * math.log(s[i]):.4f}")
        
    prob_eng_given_x += num * math.log(e[i])
    prob_sp_given_x +=  num * math.log(s[i])

#1.3
print("Q3")
prob_eng_given_x = prob_eng_given_x + math.log(prob_english)
prob_sp_given_x = prob_sp_given_x +  math.log(prob_spanish)
print(f"{prob_eng_given_x:.4f}")
print(f"{prob_sp_given_x:.4f}")


#1.4
compute =  prob_sp_given_x - prob_eng_given_x
if(compute >= 100):
    prob_eng_given_x = 0
    prob_sp_given_x = 1
elif(compute <= -100):
    prob_eng_given_x = 1
    prob_sp_given_x = 0
else:
    prob_eng_given_x = 1/(1 + math.exp(prob_sp_given_x -  prob_eng_given_x))
    prob_sp_given_x = 1 - prob_eng_given_x

print("Q4")
print(f"{prob_eng_given_x:.4f}")
