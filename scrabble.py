#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
try:
     import cPickle as pickle
except:
    import pickle

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

dictionary = {}

maxLen = 0
with open('/Users/Joseph/Documents/Puzzles/FiveThirtyEight/Scrabble/enable1.txt','r') as dictFile:
    dictLine = dictFile.readline().strip('\n')
    while dictLine:
        if len(dictLine)>maxLen:
            maxLen = len(dictLine)
        dictionary[dictLine.upper()] = sum(tiles[c]['points'] for c in dictLine.upper())
        dictLine = dictFile.readline().strip('\n')
maxLen = 15
pop_size = 150
n_steps = 100000
pnt_mut_rate = 5
swp_mut_rate = .5
blnk_mut_rate = .2
swp_blank_rate = .2
n_keep = 10

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

    # blank1_val = sum(dictionary[w]-tiles[b1]['points'] for w in blank1 if w not in words)
    # blank2_val = sum(dictionary[w]-tiles[b2]['points'] for w in blank2 if w not in words)
    # blank12_val = sum(dictionary[w]-tiles[b1]['points']-tiles[b2]['points'] for w in blank12)

    val = sum(dictionary[w] for w in words)+blank_val
    return val,words

def genetic_algorithm(init_pop,pop_size,n_steps,pnt_mut_rate,swp_mut_rate,blnk_mut_rate,swp_blank_rate,n_keep):
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
    def swp_mutate(s,swp_mut_rate):
        if np.random.rand()<swp_mut_rate:
            idx = sorted(np.random.randint(0,len(s),4))
            s = s[0:idx[0]]+s[idx[2]:idx[3]]+s[idx[1]:idx[2]]+s[idx[0]:idx[1]]+s[idx[3]:]
        return s
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
    for step in range(n_steps):
        print 'step: ' + str(step)
        new_pop = []
        score = score_pop(population)
        max_val = max(score)
        max_idx = top_idx(score)
        print 'max score: '+ str(max_val)
        best_pop = [population[i] for i in max_idx]
        # print [''.join(b[0]) for b in best_pop]
        # print [b[1] for b in best_pop]
        n_offspring = pop_size
        if n_keep>0:
            n_offspring -= len(best_pop)
        idx = reproduce(score,n_offspring)
        for i in idx:
            string = population[i][0][:]
            blanks = population[i][1][:]
            string = pnt_mutate(string,pnt_mut_rate)
            string = swp_mutate(string,swp_mut_rate)
            blanks = blnk_mutate(blanks,blnk_mut_rate)
            string,blanks = swp_blank(string,blanks,swp_blank_rate)
            new_pop.append((string,blanks))
        if n_keep>0:
            population = best_pop + new_pop
    score = score_pop(population)
    return best_pop,score,population

best_pop,score,population = genetic_algorithm(init_pop,pop_size,n_steps,pnt_mut_rate,swp_mut_rate,blnk_mut_rate,swp_blank_rate,n_keep)
output = {'best_pop':best_pop,'score':score,'population':population}
with open('scrabble_output.pickle','wb') as filename:
    pickle.dump(output,filename)
