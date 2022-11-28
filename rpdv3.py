#!/usr/bin/python3
# -v PyMuPDF
import fitz
import re
import pandas as pd
import glob
from googletrans import Translator
from clint.textui import progress
import numpy
import asyncio
from itertools import combinations

translator = Translator()

class DelimetStr:

    def __init__(self, file: list):
        self.file = file
        self.err_l = ["/", "("]
        self.f_pattern_trigger = " / nw"
        self.w_l = self.f_prepair(self.file)
        """wtf?"""
        self.pattern = r"([\w\s\-]+)\s(/)\s([\w\s][^()]+)|([\w\s]+)\s([\w()=/]+)|(\w+)"
        self.w_d = {k: self.matcher(v).__next__() for k, v in self.w_l.items()}
        # print(len(self.w_d))

    def f_prepair(self, data_list: list) -> dict:
        temp = {}
        for wf in data_list:
            if max([wf.find(err) for err in self.err_l]) > -1:
                temp[wf] = wf
            else:
                temp[wf] = wf+self.f_pattern_trigger
        return temp

    def matcher(self, w: str):
        temp = []
        m_array = re.match(self.pattern, w).groups()
        for g in m_array:
            if g and len(g) > 1 and g != "nw" and m_array.index(g) != 4:
                temp.append(g)
        yield temp


class LowerUpper:

    def __init__(self, w_list: list):
        self.w_list = w_list
        self.res = self.main()

    def main(self):
        res = list()
        for w in self.w_list:
            for f in [self.upp, self.low]:
                res.append(f(w))
        return res

    @staticmethod
    def upp(s: str) -> str:
        return s.title()

    @staticmethod
    def low(s: str) -> str:
        return s.lower()


class Template(DelimetStr):

    def __init__(self, file: list, articles: list):
        super().__init__(file)
        self.articles = {a: self.read_article(a) for a in articles}

    def read_article(self, a):
        with fitz.open(a) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text

    async def count_first(self, a_t, check_w, k_a):
        for w in check_w:
            if a_t.find(w) > -1:
                return k_a
        return None

    def set_count_loop(self, article_text):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=loop)
        tasks = [asyncio.ensure_future(self.count_first(article_text, LowerUpper(v).res, k)) for k, v in self.w_d.items()]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        return [i.result() for i in tasks if i.result()]

    def main(self) -> dict:
        container = {w: {} for w in self.w_d.keys()}
        for a, text in self.articles.items():
            print(f"Preparing: {a}")
            for k, v in progress.bar(self.w_d.items()):
                # if len(container[k]) == 0:
                for word in LowerUpper(v).res:
                    if text.find(word) > -1:
                        for s in re.findall(rf'([^.]*{word}[^.]*)', text):
                            """the most fast method #lovepy: https://overcoder.net/q/71509/%D0%B7%D0%B0%D0%BC%D0%B5%D0%BD%D0%B0-%D0%BD%D0%B5%D1%81%D0%BA%D0%BE%D0%BB%D1%8C%D0%BA%D0%B8%D1%85-%D1%81%D0%B8%D0%BC%D0%B2%D0%BE%D0%BB%D0%BE%D0%B2-%D0%BD%D0%B0-python"""
                            if a not in container[k]:
                                container[k][a] = [s.replace("\n", " ").replace(" -", "").replace("- ", "").replace("-", "") + "."]
                            else:
                                container[k][a] += [s.replace("\n", " ").replace(" -", "").replace("- ", "").replace("-", "") + "."]
                    else:
                        continue
                # else:
                #     continue
        return container

def for_print(d_w: list, sentence: str) -> str:
    position = max([sentence.find(w) for w in LowerUpper(d_w).res])
    return sentence[:position] + sentence[position:].upper()

def eng_to_ru(s: str) -> str:
    return translator.translate(s, src='en', dest='ru').text

"""get all pdf"""
words = open("words.txt", "r").read().split("\n")
# articles_list = [file for file in glob.glob("files/norm_a/*.pdf")]
articles_list = [file for file in glob.glob("files/*.pdf")]
a_c_v = input("count of articles to find words: ")
a_sets = list(combinations(articles_list, r=int(a_c_v)))
dicted_a_sets = {i: a_sets[i] for i in range(len(a_sets))}

ClassAssignment = Template(file=words, articles=articles_list)
if __name__ == "__main__":
    combinations_from_3 = {}
    """iterate again over articles already iterated over = gg, but #idontgiveafuckitspython"""
    print("searching best match...")
    for i, v in progress.bar(dicted_a_sets.items()):
        temp = []
        for var_v in v:
            matched_w = ClassAssignment.set_count_loop(ClassAssignment.articles[var_v])
            temp += list(numpy.setdiff1d(matched_w, temp))
        combinations_from_3[len(temp)] = [v, temp]
    best_match, words_m = combinations_from_3[max(combinations_from_3.keys())]
    all_searched = list(ClassAssignment.w_d.keys())
    diff = numpy.setdiff1d(all_searched, words_m)
    print("best_matches")
    [print(m) for m in best_match]
    print("Not found total: ", len(diff))
    print("This words:\n", diff)
    if input("continue y/n: ") == "y":
        fdf = list()
        """mb mk snd class? no...)"""
        AssignmentClassRes = Template(file=words, articles=best_match)
        AssignmentClass_2 = AssignmentClassRes.main().items()
        rest = len(AssignmentClass_2)
        for k, v in AssignmentClass_2:
            counter = sum([len(val) for val in v.values()])
            check_list = {}
            print("_"*20+str(rest)+"_"*20)
            shit_break = True
            for k_1, v_1 in v.items():
                if shit_break:
                    for sentence in v_1:
                        print(f"__The rest of sentences__: {str(counter)}  (Word/Frase : {k})")
                        print(for_print(AssignmentClassRes.w_d[k], sentence))
                        try:
                            check = input("add? y/b (b=break) or prev number: ")
                            # check = "y"
                        except(UnicodeDecodeError):
                            print("UnicodeDecodeError")
                        if check == "y":
                            fdf.append([k, sentence, eng_to_ru(sentence),  k_1])
                            rest -= 1
                            shit_break = False
                            break
                        elif check == "b":
                            rest -= 1
                            shit_break = False
                            break
                        elif check in check_list.keys():
                            # print([k, check_list[check][0], eng_to_ru(check_list[check][0]), check_list[check][1]])
                            fdf.append([k, check_list[check][0], eng_to_ru(check_list[check][0]), check_list[check][1]])
                            rest -= 1
                            shit_break = False
                            break
                        else:
                            check_list[str(counter)] = [sentence, k_1]
                            counter -= 1
                            continue
                else:
                    continue
        diff_before_search = numpy.setdiff1d(all_searched, list(map(lambda x : x[0], fdf)))
        fdf += [[d, "", "", ""] for d in diff_before_search]
        # print("found_words", len(set([i[0] for i in fdf])))
        """using np and pandas in two rows in one project = full bullshit, but #idontgiveafuckonmemoryitspython"""
        pd.DataFrame(fdf).to_excel("output.xlsx")
"""
need to be reFUCKtored #idontgiveafuckitworks
"""
