# -*- encoding: utf-8 -*-
import argparse
import os
import time
import pyrouge
import shutil
import codecs


def test_rouge(cand, ref_1, ref_2, ref_3):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/icsi_reference")

        references_1 = [line.strip() for line in ref_1]
        references_2 = [line.strip() for line in ref_2]
        references_3 = [line.strip() for line in ref_3]

        candidates = [line.strip() for line in cand]

        assert len(candidates) == len(references_1) == len(references_2) == len(references_3)

        cnt = len(candidates)

        for i in range(cnt):
            # model
            with open(tmp_dir + "/icsi_reference/ref.A.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references_1[i])
            with open(tmp_dir + "/icsi_reference/ref.B.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references_2[i])
            with open(tmp_dir + "/icsi_reference/ref.C.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references_3[i])

            # system
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])

        """
        Your Path
        """
        r = pyrouge.Rouge155("/users4/xiachongfeng/ROUGE-1.5.5/")

        r.model_dir = tmp_dir + "/icsi_reference/"
        r.system_dir = tmp_dir + "/candidate/"

        r.model_filename_pattern = 'ref.[A-Z].#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'

        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def rouge_results_to_str(results_dict):
    return ">> ROUGE(1/2/L): {:.2f}-{:.2f}-{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r1', type=str, default="data/icsi/icsi_reference/ref1.txt",
                        help='icsi_reference file')
    parser.add_argument('-r2', type=str, default="data/icsi/icsi_reference/ref2.txt",
                        help='icsi_reference file')
    parser.add_argument('-r3', type=str, default="data/icsi/icsi_reference/ref3.txt",
                        help='icsi_reference file')
    args = parser.parse_args()

    candidates = codecs.open(args.c, encoding="utf-8")
    references_1 = codecs.open(args.r1, encoding="utf-8")
    references_2 = codecs.open(args.r2, encoding="utf-8")
    references_3 = codecs.open(args.r3, encoding="utf-8")

    results_dict = test_rouge(candidates, references_1, references_2, references_3)

    print(rouge_results_to_str(results_dict))