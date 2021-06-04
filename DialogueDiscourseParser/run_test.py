# -*- coding: utf-8 -*-
# @Time    : 2019/9/17 10:05 AM
# @Author  : Xiachong Feng
# @File    : run_test.py
# @Software: PyCharm
import codecs
import os

if __name__ == "__main__":
    file_list = "./list.icsi"  # ./list.ami or ./list.icsi
    with codecs.open(file_list, "r", "utf-8") as f:
        names = f.readlines()
        for name in names:
            name = name.strip()
            print("*********Name:{}==={}*********".format(file_list, name))
            if name:
                os.system("CUDA_VISIBLE_DEVICES=0 python main.py --test_data={}".format(name + ".json"))
