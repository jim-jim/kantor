# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs
from os import listdir
from os.path import isfile
import pyzmail as pyz
import email, re, json
import codecs
import nltk
from nltk.corpus import treebank, stopwords
from nltk.collocations import *
from nltk.util import *
from nltk import Text
import itertools
#from nltk.book import *
def choose_text(t1, t2):
    if t1 == None or t2 == None: return ''
    if t1 == '': return t2
    if t2 == '': return t1
    if t1 == t2: return t1
    c1, c2 = 0, 0
    o1, o2 = ord('а'), ord('я')
    for ch in t1:
        if ord(ch) >= o1 and ord(ch) <= o2: c1 += 1
    for ch in t2:
        if ord(ch) >= o1 and ord(ch) <= o2: c2 += 1
    if c1 > c2: return t1
    if c1 < c2: return t2
    c1, c2 = 0, 0
    o1, o2 = ord('А'), ord('Я')
    for ch in t1:
        if ord(ch) >= o1 and ord(ch) <= o2: c1 += 1
    for ch in t2:
        if ord(ch) >= o1 and ord(ch) <= o2: c2 += 1
    if c1 > c2: return t1
    if c1 < c2: return t2
    return None
def get_charset(path):
    #that's why kantor py
    pyzm = pyz.message_from_binary_file(open(path, "rb"))
    for part in pyzm.mailparts:
        if part.charset != None: return part.charset.lower()
    try:
        st = "Content-Type:"
        with open(path, 'r') as f:
            for line in f:
                if line.startswith(st):
                    while line:
                        if line.find("1251") >= 0: return "cp1251"
                        if line.find("8-") >= 0: return "koi8-r"
                        if line.find("-8") >= 0: return "utf-8"
                        line = f.readline()
    except:
        try:
            st = "Subject:"
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith(st):
                        while line:
                            if line.find("1251") >= 0: return "cp1251"
                            if line.find("8-") >= 0: return "koi8-r"
                            if line.find("-8") >= 0: return "utf-8"
                            line = f.readline()
        except: return "utf-8"
    return "utf-8"
inp_dir = "D:\\kantor.py\\box_strelkov\\"
out_dir = "D:\\kantor.py\\box_strelkov_out\\"
inp_files = [f for f in listdir(inp_dir) if isfile(inp_dir+f)]
l = len(inp_files)
#l = 100
encs = [get_charset(inp_dir+inp_files[i]) for i in range(l)]
strs = [open(inp_dir+inp_files[i], "r", encoding=encs[i], errors="ignore").read() for i in range(l)]
#heas = [dict(re.findall(r"(?P<name>.*?):(?P<value>.*?)\n", strs[i])) for i in range(l)]
msgs = [email.message_from_string(strs[i]) for i in range(l)]
pyzs = [pyz.PyzMessage(msgs[i]) for i in range(l)]
codec = codecs.getdecoder('unicode-escape')
folders, subjects, bodies = [], [], []
for i in range(l):
    folder = str(msgs[i]["X-Yandex-FolderName"]).strip()
    subj = str(msgs[i]["Subject"])
    subj_alt = pyzs[i].get_subject()
    if (subj.startswith("=?") or subj.startswith("[SPAM]=?")) and subj.endswith("?="): _subj = subj_alt
    elif (subj_alt.startswith("=?") or subj_alt.startswith("[SPAM]=?")) and subj_alt.endswith("?="): _subj = subj
    else: _subj = choose_text(subj, subj_alt)
    htmls = [part.get_payload().decode(encs[i], errors="ignore") for part in pyzs[i].mailparts]
    htmls_alt = [codec(html)[0] for html in htmls]
    texts, texts_alt = [], []
    for j in range(len(htmls)):
        soup = bs("<html>"+htmls[j]+"</html>")
        for script in soup(["script", "style"]): script.extract()
        texts.append(soup.get_text())
    for j in range(len(htmls)):
        soup = bs("<html>"+htmls_alt[j]+"</html>")
        for script in soup(["script", "style"]): script.extract()
        texts_alt.append(soup.get_text())
    _texts = [choose_text(texts[j], texts_alt[j]) for j in range(len(texts))]
    for j in range(len(texts)):
        if _texts[j] == None:
            _texts[j] = texts[j]
    #with open(out_dir+inp_files[i]+".txt", 'w', encoding="utf-8") as f:
        #f.write(folder+"\n\n"+subj+"\n\n"+subj_alt+"\n"+"\n".join(texts)+"\n"+"\n".join(texts_alt))
        #f.write(folder+"\n\n"+_subj+"\n"+"\n".join(texts)+"\n"+"\n".join(texts_alt))
        #f.write(folder+"\n\n"+_subj+"\n"+"\n".join(_texts))
    folders.append(folder)
    subjects.append(_subj)
    bodies.append(_texts)
#docs = [nltk.word_tokenize(item) for item in itertools.chain(*bodies)]
docs = [[jtem.lower() for jtem in nltk.word_tokenize(item)] for item in itertools.chain(*bodies)]
words = itertools.chain(*docs)
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words, 2)
finder.apply_freq_filter(10)
ignored_words = stopwords.words('russian')
ignored_words_en = stopwords.words('english')
#word_filter = lambda w: len(w) < 3 or w.lower() in ignored_words or w[0] in ['.', '/']
word_filter = lambda w: len(w) < 3 or w in ignored_words or w[0] in ['.', '/']
finder.apply_word_filter(word_filter)
#colloc = finder.nbest(bigram_measures.likelihood_ratio, 1000)
colloc = finder.nbest(bigram_measures.pmi, 1000)
with open(out_dir+"_.txt", 'w', encoding="utf-8") as f:
    f.write("\n".join(' '.join(item) for item in colloc))