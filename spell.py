from collections import Counter
import re
import numpy as np

def words(text): return re.findall(r'\w+', text)

corpus = open('corpus.txt').read().lower()
WORDS = Counter(words(corpus))
letters 	= 'abcdefghijklmnopqrstuvxyz?.\'-_'

def P(word, N=sum(WORDS.values())):
	return WORDS[word] / N

def correction1(word):
	#print("hele")
	return max(candidates(word), key=P)

def candidates(word):
	return known(edits1(word)) or known([word])

def known(words):
	return set(w for w in words if w in WORDS)

# Constructs the words whose edit distance is 1 to given word.
def edits1(word):
	splits 		= [ (word[:i], word[i:]) 	for i in range(len(word) + 1)]
	deletes 	= [ L + R[1:] 			 	for L, R in splits if R]
	transposes 	= [ L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
	replaces    = [ L + c + R[1:]			for L, R in splits if R for c in letters]
	inserts		= [ L + c + R    			for L, R in splits for c in letters]
	return set(deletes + transposes + replaces + inserts)

#print(correction1(input()))


##########   NOISY CHANNEL   ########## 


# Constructs the Damerau-Levensthein edit table to find the edit distance.
# Returns the table of operations for 2 words.
def DL_edit(word1, word2):
	dp = [[0 for j in range(len(word2)+1)] for i in range(len(word1)+1)]
	ops = [["none" for j in range(len(word2)+1)] for i in range(len(word1)+1)]

	for i in range(len(word1)+1):
		dp[i][0] = i;
		ops[i][0] = "del"

	for i in range(len(word2)+1):
		dp[0][i] = i;
		ops[0][i] = "ins"

	ops[0][0] = "none"

	for i in range(1, len(word1)+1):
		op = "none"
		for j in range(1, len(word2)+1):
			n = min( (dp[i-1][j]+1), (dp[i][j-1]+1), dp[i-1][j-1]+int(word1[i-1] != word2[j-1]) )
			if n == dp[i-1][j]+1: #insertion
				op = "del"
			elif n == dp[i][j-1]+1: #deletion
				op = "ins"
			elif n == dp[i-1][j-1]+1: #substitution
				op = "sub"
			elif n == dp[i-1][j-1]: #none
				op = "diag"
			if (word1[i-1]==word2[j-2] and word1[i-2]==word2[j-1]) and (dp[i-2][j-2]+1)<n:
				n = dp[i-2][j-2]+1
				op = "trans"
			dp[i][j] = n
			ops[i][j] = op

	#print(ops)
	return ops

# Traces back through the edit table, returns the character on first edit
def traceBack(ops, word1, word2):
	i = len(ops)-1
	j = len(ops[0])-1
	op = ops[i][j]

	while (i>=0 and j>=0) and op != "none":

		op = ops[i][j]
		#print(op)
		ch1 = ''
		ch2 = ''
		m=l-1
		n=l-1

		if i>0 :
			ch1 = word1[i-1]
			if ch1 == '?' : m = l-6
			elif ch1 == '.' : m = l-5
			elif ch1 == '\'' : m = l-4
			elif ch1 == '-' : m = l-3
			elif ch1 == '_' : m = l-2
			else : m = ord(ch1) - ord('a')
		
		if j>0 : 
			ch2 = word2[j-1]
			if ch2 == '?' : n = l-6
			elif ch2 == '.' : n = l-5
			elif ch2 == '\'' : n = l-4
			elif ch2 == "-" : n = l-3
			elif ch2 == "_" : n = l-2
			else : n = ord(ch2) - ord('a')

		#print(ch1, " ", ch2)
		#print(m, " ", n)

		if op == "diag" and ch1==ch2:
			i-=1
			j-=1
		else:
			return [op, ch1, ch2]

# Confusion matrices indiced from a-z and special characters: ? . ' - _ and blank character
# Rows hold the correct letter, columns hold the error
l = ord('z')-ord('a') + 7
DEL = [[1 for j in range(l)] for i in range(l)]
INS = [[1 for j in range(l)] for i in range(l)]
SUB = [[1 for j in range(l)] for i in range(l)]
TRANS = [[1 for j in range(l)] for i in range(l)]

# Traces back through the edit table while adding the operations to the confusion matrix.
def traceBackAndFill(ops, word1, word2):
	i = len(ops)-1
	j = len(ops[0])-1
	op = ops[i][j]

	while (i>=0 and j>=0) and op != "none":

		op = ops[i][j]
		#print(op)
		ch1 = ''
		ch2 = ''
		m=l-1
		n=l-1

		if i>0 :
			ch1 = word1[i-1]
			if ch1 == '?' : m = l-6
			elif ch1 == '.' : m = l-5
			elif ch1 == '\'' : m = l-4
			elif ch1 == '-' : m = l-3
			elif ch1 == '_' : m = l-2
			else : m = ord(ch1) - ord('a')
		
		if j>0 : 
			ch2 = word2[j-1]
			if ch2 == '?' : n = l-6
			elif ch2 == '.' : n = l-5
			elif ch2 == '\'' : n = l-4
			elif ch2 == "-" : n = l-3
			elif ch2 == "_" : n = l-2
			else : n = ord(ch2) - ord('a')

		#print(ch1, " ", ch2)
		#print(m, " ", n)

		if op == "diag" :
			i-=1
			j-=1
		elif op == "ins" :
			INS[m][n] += 1
			j-=1
		elif op == "del" :
			DEL[m][n] += 1
			i-=1
		elif op == "sub" :
			SUB[m][n] +=1
			i-=1
			j-=1
		elif op == "trans" :
			TRANS[m][n] += 1
			i-=2
			j-=2

#k = DL_edit("jamjar", "jam_jar")
#print("OPS:",np.matrix(k))
#print(l)
#traceBack(k,"jamjar", "jam_jar")
#print("INS:",np.matrix(INS))

def fillConf(word1, word2):
	ops = DL_edit(word1, word2)
	traceBackAndFill(ops, word1, word2)

# Reads the spell-error.txt file to fill up the confusion matrix.
def createConf():
	reg = re.compile('(.*)\*(\d+)')
	with open("spell-errors.txt") as f:
		for line in f:
			line = line.lower()
			arr = line.replace(':', ' ').replace(',', ' ').split()
			word1 = arr.pop(0)
			for el in arr:
				word2 = el
				k = reg.match(el) # check if the error is made more than once
				if k!= None:
					s = k.group(1)
					word2 = s
					for i in range(int(k.group(2))-1):
						arr.append(k.group(1))
				fillConf(word1, word2)
			#print(arr)

createConf()
#print("INS:",np.matrix(INS))
#print("INS:",np.matrix(DEL))
#print("INS:",np.matrix(TRANS))
#print("INS:",np.matrix(SUB))


def normalizeConf():
	for i, ch1 in letters:
		for j, ch2 in letters:
			reg1 = len(re.findall(edit[1], corpus))
			reg2 = len(re.findall(edit[1]+edit[2], corpus))
			INS[i][j] = INS[i][j]+1 / (reg1 + l)
			SUB[i][j] = SUB[i][j]+1/ ( reg1 + l) 
			DEL[i][j] = DEL[i][j]+1 / (reg2 + l)  
			TRANS[i][j] = TRANS[i][j]+1 / (reg2 + l) 

# Correction with noisy channel model
def correction2(word):
	word1List = edits1(word)
	word2 = word
	maxP = 0;
	res = "";
	for _,e in enumerate(word1List):
		word1 = e
		edit = traceBack(DL_edit(word1, word2), word1, word2)
		op = edit[0]
		ch1 = l-1
		ch2 = l-1
		likeli = 0

		if edit[1] == '?' : ch1 = l-6
		elif edit[1] == '.' : ch1 = l-5
		elif edit[1] == '\'' : ch1 = l-4
		elif edit[1] == '-' : ch1 = l-3
		elif edit[1] == '_' : ch1 = l-2
		elif edit[1]!= "": ch1 = ord(edit[1])-ord('a')

		if edit[2] == '?' : ch2 = l-6
		elif edit[2] == '.' : ch2 = l-5
		elif edit[2] == '\'' : ch2 = l-4
		elif edit[2] == '-' : ch2 = l-3
		elif edit[2] == '_' : ch2 = l-2
		elif edit[2]!= "": ch2 = ord(edit[2])-ord('a')

		if op == "ins":
			likeli = INS[ch1][ch2] 
		elif op == "del":
			likeli = DEL[ch1][ch2]
		elif op == "trans":
			likeli = TRANS[ch1][ch2]
		elif op == "sub":
			likeli = SUB[ch1][ch2]

		if likeli <0:
			print(word)

		prioP = likeli* P(word1) # Noisy channel model
		#prioP = 1* P(word1) # Language model
		if prioP > maxP: res = word1

	print(word+ ": "+ res)
	return res

missWords = words(open('test-words-misspelled.txt').read().lower())
corrWords = words(open('test-words-correct.txt').read().lower())

def testCorrection(missWords, corrWords):
	t = 0
	f = 0
	for i in range(len(missWords)):
		corr = correction2(missWords[i])
		if corr!=corrWords[i] : f+=1
		else: t+=1
		if corr=="": f-=1

	print(t,f)
	print(t/(t+f))

testCorrection(missWords, corrWords)
#print(missWords[0])


