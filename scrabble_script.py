#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import time
try:
     import cPickle as pickle
except:
    import pickle


# Get the values of all the tiles. Blanks treated separately as '1' and '2'

# In[15]:


tiles = {}
tiles['A'] = {'count':9,'points':1}
tiles['B'] = {'count':2,'points':3}
tiles['C'] = {'count':2,'points':3}
tiles['D'] = {'count':4,'points':2}
tiles['E'] = {'count':12,'points':1}
tiles['F'] = {'count':2,'points':4}
tiles['G'] = {'count':3,'points':2}
tiles['H'] = {'count':2,'points':4}
tiles['I'] = {'count':9,'points':1}
tiles['J'] = {'count':1,'points':8}
tiles['K'] = {'count':1,'points':5}
tiles['L'] = {'count':4,'points':1}
tiles['M'] = {'count':2,'points':3}
tiles['N'] = {'count':6,'points':1}
tiles['O'] = {'count':8,'points':1}
tiles['P'] = {'count':2,'points':3}
tiles['Q'] = {'count':1,'points':10}
tiles['R'] = {'count':6,'points':1}
tiles['S'] = {'count':4,'points':1}
tiles['T'] = {'count':6,'points':1}
tiles['U'] = {'count':4,'points':1}
tiles['V'] = {'count':2,'points':4}
tiles['W'] = {'count':2,'points':4}
tiles['X'] = {'count':1,'points':8}
tiles['Y'] = {'count':2,'points':4}
tiles['Z'] = {'count':1,'points':10}
tiles['1'] = {'count':1,'points':0}
tiles['2'] = {'count':1,'points':0}


# load dictionary from enable1.txt and get values of each word in tiles

# In[16]:


maxLen = 0
dictionary = {}
with open('/Users/Joseph/Documents/Puzzles/FiveThirtyEight/Scrabble/enable1.txt','r') as dictFile:
    dictLine = dictFile.readline().strip('\n')
    while dictLine:
        if len(dictLine)>maxLen:
            maxLen = len(dictLine)
        dictionary[dictLine.upper()] = sum(tiles[c]['points'] for c in dictLine.upper())
        dictLine = dictFile.readline().strip('\n')


# set params

# In[17]:


maxLen = 15 # max word len to check - override longest in dict - possibly still too long
pop_size = 150
n_steps = 100000
pnt_mut_rate = 4 # expected number of direct character swaps
ins_mut_rate = 1 # expected number of substring relocations
blnk_mut_rate = .2 # probability of blank randomly changing to any other value
swp_blank_rate = .2 # probability of blank swapping with matching character
n_keep = 5 # number from top of list surviving to next generation


# generate 'pop_size' random starting strings from the tiles, assign random values to blanks.

# In[18]:


def gen_rand_string(n):
    strings = []
    base_string = ''.join(k*tiles[k]['count'] for k in tiles.keys())
    base_list = list(base_string)
    for i in range(n):
        x = random.sample(base_list,len(base_list))
        strings.append(x)
    return strings

strs = gen_rand_string(pop_size)
init_pop = []
for s in strs:
    blanks = [chr(c) for c in np.random.randint(65,91,2)]
    init_pop.append((s,blanks))


# define evaluation function. substrings without blanks are easy to check against dictionary. with blanks, sub in blank value, check against existing word lists and dictionary, get value, and subtract value of tile replaced by blank.

# In[19]:


def eval_string(string,b1,b2,maxLen):
    words = set()
    blank1 = set()
    blank2 = set()
    blank12 = set()
    for i in range(len(string)):
        for j in range(min(maxLen,len(string)-i)):
            word = ''.join(string[i:i+j])
            if '1' in word:
                if '2' in word:
                    word.replace('1',b1)
                    word.replace('2',b2)
                    if word in dictionary:
                        blank12.add(word)
                else:
                    word.replace('1',b1)
                    if word in dictionary:
                        blank1.add(word)
            elif '2' in word:
                word.replace('2',b2)
                if word in dictionary:
                    blank2.add(word)
            elif word in dictionary:
                words.add(word)
    blank_dict = {w:dictionary[w]-tiles[b1]['points'] for w in blank1 if w not in words}
    for w in blank2:
        if w not in words:
            val = dictionary[w]-tiles[b2]['points']
            if w not in blank_dict:
                blank_dict[w] = val
            elif blank_dict[w]<val:
                blank_dict[w] = val
    for w in blank12:
        if (w not in words):
            val = dictionary[w]-tiles[b1]['points']-tiles[b2]['points']
            if w not in blank_dict:
                blank_dict[w] = val
            elif blank_dict[w]<val:
                blank_dict[w] = val

    blank_val = sum(blank_dict.values())

    val = sum(dictionary[w] for w in words)+blank_val
    return val,words


# define the genetic algorithm. use the evaluation function to weight strings in population, then use series of mutations to evolve population over iterations.

# In[20]:


def genetic_algorithm(init_pop,pop_size,n_steps,pnt_mut_rate,blnk_mut_rate,swp_blank_rate,ins_mut_rate,n_keep):
    def reproduce(scores,k):
        scores = np.array(scores,dtype='float')
        scores -= min(scores)
        scores /= sum(scores)
        bins = np.append(0,np.cumsum(scores))
        r = np.random.rand(k)
        idx = np.digitize(r,bins)-1
        return idx
    def pnt_mutate(string,pnt_mut_rate):
        n = len(string)
        p = pnt_mut_rate/float(n)
        r = np.random.rand(n)
        for i in range(n):
            if r[i] < p:
                swp_idx = np.random.randint(n)
                swp = string[i]
                string[i] = string[swp_idx]
                string[swp_idx] = swp
        return string
    def blnk_mutate(blanks,blnk_mutate_rate):
        for i in range(len(blanks)):
            if np.random.rand()<blnk_mutate_rate:
                blanks[i] = chr(np.random.randint(65,91))
        return blanks
    def swp_blank(string,blanks,swp_blank_rate):
        for i in ['1','2']:
            if np.random.rand()<swp_blank_rate:
                b = blanks[int(i)-1]
                idx = [x for x in range(len(string)) if string[x]==b]
                idx = np.random.choice(idx)
                blank_idx = string.index(i)
                string[blank_idx] = b
                string[idx] = i
        return string,blanks
    def ins_mutate(string,ins_mut_rate):
        n_swps = np.random.poisson(ins_mut_rate)
        for i in range(n_swps):
            n = len(string)
            idx = np.random.randint(0,len(string),2)
            l = np.random.randint(0,min(12,1+(idx[1]-idx[0])%n,1+(n-idx[0])))
            tmp = string[idx[0]:idx[0]+l]
            if idx[1]>idx[0]:
                string[idx[0]:idx[1]-l] = string[idx[0]+l:idx[1]]
                string[idx[1]-l:idx[1]] = tmp
            else:
                string[idx[1]+l:idx[0]+l] = string[idx[1]:idx[0]]
                string[idx[1]:idx[1]+l] = tmp
        return string
    def score_pop(population):
        score = []
        for indiv in population:
            string = indiv[0]
            b1 = indiv[1][0]
            b2 = indiv[1][1]
            score.append(eval_string(string,b1,b2,maxLen)[0])
        return score
    def top_idx(scores):
        idx = sorted(range(len(scores)),key=lambda i: scores[i])[-n_keep:]
        return idx

    population = init_pop
    max_score = []
    start = time.time()
    for step in range(n_steps):
        new_pop = []
        score = score_pop(population)
        max_val = max(score)
        max_idx = top_idx(score)
        max_score.append(max_val)
        best_pop = [population[i] for i in max_idx]
        n_offspring = pop_size
        if n_keep>0:
            n_offspring -= len(best_pop)
        idx = reproduce(score,n_offspring)
        if step%1000==0:
            t = time.time()
            print 'step: ' + str(step)
            print 'max score: '+ str(max_val)
            print 'string: ' + ''.join(best_pop[-1][0])
            print 'blanks: ["'+best_pop[-1][1][0]+'","'+best_pop[-1][1][1]+'"]'
            print 'time: '+str(t-start)
            print ''
        for i in idx:
            string = population[i][0][:]
            blanks = population[i][1][:]
            string = pnt_mutate(string,pnt_mut_rate)
            blanks = blnk_mutate(blanks,blnk_mut_rate)
            string,blanks = swp_blank(string,blanks,swp_blank_rate)
            string = ins_mutate(string,ins_mut_rate)
            new_pop.append((string,blanks))
        if n_keep>0:
            population = best_pop + new_pop
    score = score_pop(population)
    return best_pop,score,population,max_score


# run this thing and pickle it

# In[21]:


best_pop,score,population,max_score = genetic_algorithm(init_pop,pop_size,n_steps,pnt_mut_rate,blnk_mut_rate,swp_blank_rate,ins_mut_rate,n_keep)

output = {'best_pop':best_pop,'score':score,'population':population,'max_score':max_score}
time = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
file_name = 'scrabble_output_'+time+'.pickle'
with open(file_name,'wb') as f:
    pickle.dump(output,f)


# get pickled output and see.

# In[22]:


with open(file_name,'rb') as f:
    output = pickle.load(f)
for b in output['best_pop']:
    val,words = eval_string(b[0],b[1][0],b[1][1],maxLen)
    print 'score: ' + str(val)
    print 'n_words: ' + str(len(words))
    print 'string: ' + ''.join(b[0])
    print 'blank_1: ' + b[1][0]
    print 'blank_2: ' + b[1][1]


# In[23]:


plt.figure()
plt.plot(range(n_steps),output['max_score'])
plt.xlabel('generation')
plt.ylabel('best score')
