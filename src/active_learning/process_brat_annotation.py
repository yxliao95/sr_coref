import ast
import logging
import os
import shutil
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd

from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.file_checker import FileChecker

FILE_CHECKER = FileChecker()
INCLUDE_SINGLETON = False

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


class AnnMentionClass:
    def __init__(self) -> None:
        self.id = ""
        self.type = "Mention"
        self.start_index = ""
        self.end_index = ""
        self.mention_str = ""

    def get_ann_str(self) -> str:
        return f"{self.id}\t{self.type} {self.start_index} {self.end_index}\t{self.mention_str}\n"

    def set_end_index(self, value, text):
        self.end_index = value
        self.mention_str = text[self.start_index : self.end_index]

    def __repr__(self) -> str:
        return self.get_ann_str()

    def __str__(self) -> str:
        return self.get_ann_str()


class AnnCoreferenceClass:
    def __init__(self) -> None:
        self.id = ""
        self.type = "Coreference"
        self.anaphora = ""
        self.antecedent = ""

    def get_ann_str(self) -> str:
        return f"{self.id}\t{self.type} Anaphora:{self.anaphora} Antecedent:{self.antecedent}\t\n"

    def __repr__(self) -> str:
        return self.get_ann_str()

    def __str__(self) -> str:
        return self.get_ann_str()


def get_AnnMentionClass_notClosed(ann_ment_list: list[AnnMentionClass]) -> AnnMentionClass:
    for annMent in ann_ment_list:
        if annMent.end_index == "":
            return annMent
    return None


def prepare_brat(config, sampled_doc_dict: dict):
    brat_output_dir = config.output.brat.unfinished_dir
    check_and_remove_dirs(brat_output_dir, True)

    csv_source_base_dir = config.output.temp.model_inference_csv_dir
    token_colName = config.name_style.fastcoref_joint.column_name.token_from_spacy
    sentence_group_colName = config.name_style.fastcoref_joint.column_name.sentence_group
    coref_group_conll_colName = config.name_style.fastcoref_joint.column_name.coref_group_conll

    for section_name, doc_list in sampled_doc_dict.items():
        for doc_name in doc_list:
            doc_id = doc_name.rstrip(".csv")
            df_spacy = pd.read_csv(
                os.path.join(csv_source_base_dir, section_name, doc_name), index_col=0, na_filter=False
            )
            df_sentence = df_spacy.groupby([sentence_group_colName])[token_colName].apply("#@#".join).reset_index()
            df_coref = df_spacy.groupby([sentence_group_colName])[coref_group_conll_colName].apply(list).reset_index()

            sentences = [
                str(_series.get(token_colName)).replace("#@#", " ").strip() for _, _series in df_sentence.iterrows()
            ]
            text = "\n".join(sentences)

            # .txt files
            output_dir = os.path.join(brat_output_dir, section_name)
            check_and_create_dirs(output_dir)
            with open(os.path.join(output_dir, f"{doc_id}.txt"), "w", encoding="UTF-8") as f:
                f.write(text)

            # .ann files
            with open(os.path.join(output_dir, f"{doc_id}.ann"), "w", encoding="UTF-8") as f:
                mention_id = 0
                groupNum_mentions_dict: dict[int, list[AnnMentionClass]] = defaultdict(list)
                offset = 0
                for _idx, _series in df_sentence.iterrows():
                    token_list = (
                        _series.get(token_colName).strip().strip("#@#").split("#@#")
                    )  # Remove the last whitespace
                    conll_labelStr_list = df_coref.loc[_idx,].get(
                        coref_group_conll_colName
                    )  # The corresponding conll label of last whitespace are reamined so far

                    for tok_id, tok in enumerate(
                        token_list
                    ):  # The corresponding conll label of last whitespace will be ignored
                        conll_labelListStr = conll_labelStr_list[tok_id]
                        if conll_labelListStr not in [-1, "-1", np.nan]:
                            for conll_label in ast.literal_eval(conll_labelListStr):
                                if "(" in conll_label:
                                    ann_mention_class = AnnMentionClass()
                                    ann_mention_class.id = f"T{mention_id}"
                                    ann_mention_class.start_index = offset
                                    mention_id += 1
                                    if ")" in conll_label:
                                        ann_mention_class.set_end_index(offset + len(tok), "\n".join(sentences))
                                        coref_id = int(conll_label.replace("(", "").replace(")", ""))
                                    else:
                                        coref_id = int(conll_label.replace("(", ""))
                                    groupNum_mentions_dict[coref_id].append(ann_mention_class)
                                elif "(" not in conll_label and ")" in conll_label:
                                    coref_id = int(conll_label.replace(")", ""))
                                    ann_mention_class = get_AnnMentionClass_notClosed(groupNum_mentions_dict[coref_id])
                                    ann_mention_class.set_end_index(offset + len(tok), "\n".join(sentences))

                        offset += len(tok) + 1
                    offset = (
                        sum([len(sent) for sent in sentences[0 : _idx + 1]]) + _idx + 1
                    )  # The offset of the sentence start.

                pair_id = 0
                for _, ann_mention_list in groupNum_mentions_dict.items():
                    for _id, ann_mention_class in enumerate(ann_mention_list):
                        f.write(ann_mention_class.get_ann_str())

                    for _id, ann_mention_class in enumerate(ann_mention_list):
                        if _id == 0:
                            continue
                        ann_coref_class = AnnCoreferenceClass()
                        ann_coref_class.id = f"R{pair_id}"
                        ann_coref_class.anaphora = ann_mention_list[_id - 1].id
                        ann_coref_class.antecedent = ann_mention_list[_id].id

                        f.write(ann_coref_class.get_ann_str())
                        pair_id += 1

    return brat_output_dir


class BratMention:
    def __init__(self, id, start, end, mention_str) -> None:
        self.id = id
        self.tok_start = start
        self.tok_end = end  # Not inclusive
        self.group_id_list = (
            []
        )  # Will only have one element, as brat ann scheme not allow to assign one mention to multi coref cluster (for now)
        self.mention_str = mention_str

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BratMention):
            return self.id == __o.id
        else:
            return self.id == __o

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.id}({self.tok_start},{self.tok_end})"


class BratCorefGroup:
    def __init__(self) -> None:
        self.coref_group: list[set[BratMention]] = []

    def add(self, ment_a: BratMention, ment_b: BratMention):
        to_be_merged = set()
        for group_id, mention_set in enumerate(self.coref_group):
            if ment_a in mention_set or ment_b in mention_set:
                to_be_merged.add(group_id)
                mention_set.update([ment_a, ment_b])
        if len(to_be_merged) == 0:
            # Not exist in curr group, thus create new group
            self.coref_group.append({ment_a, ment_b})
        elif len(to_be_merged) > 1:
            # Exist in multiple groups, thus need to merge
            new_group = set()
            to_be_removed = list(to_be_merged.copy())
            while to_be_merged:
                new_group = new_group.union(self.coref_group[to_be_merged.pop()])
            for index in sorted(to_be_removed, reverse=True):
                del self.coref_group[index]
            self.coref_group.append(new_group)

    def __str__(self) -> str:
        out = []
        for group in self.coref_group:
            out.append(",".join(map(str, group)))
        return "|".join(out)


def find_sub_list(sublist, source_list) -> tuple[int, int]:
    """Returns: start index, end index (inclusive)"""
    sll = len(sublist)
    for ind in (i for i, e in enumerate(source_list) if e == sublist[0]):
        if source_list[ind : ind + sll] == sublist:
            return ind, ind + sll - 1


def resolve_brat(config):
    brat_source_dir = config.output.brat.finished_dir
    unlabeled_pool_dir = config.unlabeled_pool.base_dir
    output_base_dir = config.output.temp.manual_annotation_csv_dir

    spacy_nametyle = config.name_style.spacy.column_name
    gt_namestyle = config.name_style.mimic_cxr_gt.column_name

    for section_name in ["findings", "impression"]:
        brat_dir = os.path.join(brat_source_dir, section_name)
        if not os.path.exists(brat_dir):
            continue
        spacy_dir = os.path.join(unlabeled_pool_dir, section_name)
        for doc_id in [f.rstrip(".txt") for f in FILE_CHECKER.filter(os.listdir(brat_dir)) if ".txt" in f]:
            # brat outputs
            with open(os.path.join(brat_dir, doc_id + ".txt"), "r", encoding="UTF-8") as f:
                txt_file_str = "".join(f.readlines())
            with open(os.path.join(brat_dir, doc_id + ".ann"), "r", encoding="UTF-8") as f:
                ann_file_list = f.readlines()
            # The source of the brat txt files.
            df_spacy = pd.read_csv(os.path.join(spacy_dir, f"{doc_id}.csv"), index_col=0, na_filter=False)
            # Sometime a token is whitespace, which would make the split() not work as expecting. Thus we use other symbol
            df_sentence = (
                df_spacy.groupby([spacy_nametyle.sentence_group])[spacy_nametyle.token].apply("#@#".join).reset_index()
            )
            sentences_withoutstrip = [str(_series.get(spacy_nametyle.token)) for _, _series in df_sentence.iterrows()]
            sentence_tok_id = [arr for key, arr in df_spacy.groupby([spacy_nametyle.sentence_group]).indices.items()]

            # Align to spacy. Also read the brat token offset.
            idx = 0
            brat_offset = 0
            df_brat = pd.DataFrame(columns=["brat_tok", "spacy_index", "brat_offset"])
            for sent_id, (sentence_str, id_list) in enumerate(zip(sentences_withoutstrip, sentence_tok_id)):
                tok_list_spacy = sentence_str.split("#@#")
                tok_list_brat = (
                    sentence_str.strip().strip("#@#").strip("").split("#@#")
                )  # When generating brat txt, whitespaces are stripped.

                if len(tok_list_brat) == 1 and tok_list_brat[0] == "":
                    start, end = 0, 0
                else:
                    start, end = find_sub_list(tok_list_brat, tok_list_spacy)

                prev_brat_tok = ""
                for brat_tok, spacy_idx in zip(tok_list_brat, id_list[start : end + 1]):
                    brat_offset = (
                        brat_offset
                        + len(prev_brat_tok)
                        + txt_file_str[brat_offset + len(prev_brat_tok) :].index(brat_tok)
                    )
                    if not df_brat.empty and brat_offset == df_brat.iloc[-1]["brat_offset"]:
                        brat_offset += 1  # In case brat_tok is "". And doing this do not affect the next tok
                    df_brat.loc[idx] = (brat_tok, spacy_idx, brat_offset)
                    idx += 1
                    prev_brat_tok = brat_tok

            df_aligned = (
                df_spacy.merge(df_brat, how="outer", left_index=True, right_on="spacy_index")
                .reset_index()
                .drop(columns=["index"])
            )
            df_aligned = df_aligned.loc[
                :,
                [
                    spacy_nametyle.token,
                    spacy_nametyle.token_offset,
                    spacy_nametyle.sentence_group,
                    "brat_tok",
                    "spacy_index",
                    "brat_offset",
                ],
            ]
            df_aligned[gt_namestyle.coref_group] = [-1] * len(df_aligned)
            df_aligned[gt_namestyle.coref_group_conll] = [-1] * len(df_aligned)

            # Resolve brat files
            mention_list: list[BratMention] = []
            brat_coref_obj = BratCorefGroup()
            # Create two loop for mention and relation respectively, in case they are not placed in order
            for line in [line.strip() for line in ann_file_list]:
                line_info_list = line.split("\t")
                # print(line_info_list)
                if line[0] == "T":
                    # Mention
                    mention_id = line_info_list[0]
                    ment_start = line_info_list[1].split(" ")[1]
                    ment_end = line_info_list[1].split(" ")[-1]
                    mention_str = line_info_list[2]
                    mention_list.append(BratMention(mention_id, ment_start, ment_end, mention_str))
            for line in [line.strip() for line in ann_file_list]:
                line_info_list = line.split("\t")
                if line[0] == "R":
                    # relation
                    relation_id = line_info_list[0]
                    mention_a_id = line_info_list[1].split(" ")[1].split(":")[-1]
                    mention_b_id = line_info_list[1].split(" ")[2].split(":")[-1]
                    mention_a = mention_list[mention_list.index(mention_a_id)]
                    mention_b = mention_list[mention_list.index(mention_b_id)]
                    brat_coref_obj.add(mention_a, mention_b)

            # Assign coref group id to mentions
            for coref_id, coref_group in enumerate(brat_coref_obj.coref_group):
                for mention in coref_group:
                    mention.group_id_list.append(coref_id)

            # Assign coref group id to singleton mention
            all_coreferent_mention = [mention for coref_group in brat_coref_obj.coref_group for mention in coref_group]
            next_coref_id = len(brat_coref_obj.coref_group)
            for mention in mention_list:
                if mention not in all_coreferent_mention:
                    mention.group_id_list.append(next_coref_id)
                    next_coref_id += 1

            # Put conll labels into df
            if INCLUDE_SINGLETON:
                source_mention_list = mention_list
            else:
                source_mention_list = all_coreferent_mention

            for mention in source_mention_list:
                row_condition = (df_aligned["brat_offset"] >= int(mention.tok_start)) & (
                    df_aligned["brat_offset"] < int(mention.tok_end)
                )
                # df_aligned.loc[row_condition, gt_namestyle.coref_group] = int(coref_id)
                target_rows = df_aligned.loc[row_condition, gt_namestyle.coref_group]
                mention_str = ""
                if len(target_rows) == 1:  # mention has only one token
                    target_idx = df_aligned.loc[row_condition].iloc[0].name
                    if df_aligned.loc[target_idx, gt_namestyle.coref_group] == -1:
                        df_aligned.loc[target_idx, gt_namestyle.coref_group] = str(mention.group_id_list)
                        df_aligned.loc[target_idx, gt_namestyle.coref_group_conll] = str(
                            [f"({coref_id})" for coref_id in mention.group_id_list]
                        )
                    else:
                        # Append new element to exiting list
                        group_id_list = ast.literal_eval(df_aligned.loc[target_idx, gt_namestyle.coref_group])
                        group_id_list.extend(mention.group_id_list)
                        df_aligned.loc[target_idx, gt_namestyle.coref_group] = str(list(set(group_id_list)))

                        group_conll_str_list = ast.literal_eval(
                            df_aligned.loc[target_idx, gt_namestyle.coref_group_conll]
                        )
                        group_conll_str_list.extend([f"({coref_id})" for coref_id in mention.group_id_list])
                        df_aligned.loc[target_idx, gt_namestyle.coref_group_conll] = str(group_conll_str_list)

                    mention_str = " ".join(df_aligned.loc[row_condition].get(spacy_nametyle.token).to_list())
                elif len(target_rows) > 1:  # mention has more than one token
                    # coref_group
                    for index, row_series in df_aligned.loc[row_condition].iterrows():
                        if row_series.loc[gt_namestyle.coref_group] == -1:
                            df_aligned.loc[index, gt_namestyle.coref_group] = str(mention.group_id_list)
                        else:
                            # Append new element to exiting list
                            group_id_list = ast.literal_eval(row_series.loc[gt_namestyle.coref_group])
                            group_id_list.extend(mention.group_id_list)
                            df_aligned.loc[index, gt_namestyle.coref_group] = str(list(set(group_id_list)))

                    # coref_group_conll
                    first_idx = df_aligned.loc[row_condition].iloc[0].name
                    last_idx = df_aligned.loc[row_condition].iloc[-1].name

                    if df_aligned.loc[first_idx, gt_namestyle.coref_group_conll] == -1:
                        df_aligned.loc[first_idx, gt_namestyle.coref_group_conll] = str(
                            [f"({coref_id}" for coref_id in mention.group_id_list]
                        )
                    else:
                        group_conll_str_list = ast.literal_eval(
                            df_aligned.loc[first_idx, gt_namestyle.coref_group_conll]
                        )
                        group_conll_str_list.extend([f"({coref_id}" for coref_id in mention.group_id_list])
                        df_aligned.loc[first_idx, gt_namestyle.coref_group_conll] = str(group_conll_str_list)

                    if df_aligned.loc[last_idx, gt_namestyle.coref_group_conll] == -1:
                        df_aligned.loc[last_idx, gt_namestyle.coref_group_conll] = str(
                            [f"{coref_id})" for coref_id in mention.group_id_list]
                        )
                    else:
                        group_conll_str_list = ast.literal_eval(
                            df_aligned.loc[last_idx, gt_namestyle.coref_group_conll]
                        )
                        group_conll_str_list.extend([f"{coref_id})" for coref_id in mention.group_id_list])
                        df_aligned.loc[last_idx, gt_namestyle.coref_group_conll] = str(group_conll_str_list)

                    mention_str = " ".join(df_aligned.loc[first_idx:last_idx].get(spacy_nametyle.token).to_list())

                try:
                    assert mention_str == mention.mention_str
                except AssertionError:
                    logger.warning(
                        "AssertionError warning: doc_id: %s, brat label: [%s], spacy token: [%s]",
                        doc_id,
                        mention.mention_str,
                        mention_str,
                    )
                    # raise err
            # display(HTML(df_aligned.to_html()))

            # Write CSV files
            output_dir = os.path.join(output_base_dir, section_name)
            check_and_create_dirs(output_dir)
            df_out = df_aligned.loc[
                :,
                [
                    spacy_nametyle.token,
                    spacy_nametyle.sentence_group,
                    gt_namestyle.coref_group,
                    gt_namestyle.coref_group_conll,
                ],
            ]
            # display(HTML(df_out.to_html()))
            df_out.to_csv(os.path.join(output_dir, f"{doc_id}.csv"))

        logger.info(output_dir)


def copy_brat_configs(src_dir, dst_dir):
    file_list = os.listdir(src_dir)
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        shutil.copy2(src_file, dst_file)


@hydra.main(version_base=None, config_path=config_path, config_name="active_learning")
def main(config):
    # Generate brat ann data
    # with open(config.output.log.labeled_pool_info_file, "r", encoding="utf-8") as f:
    #     lines = f.readlines()

    # sampled_doc_dict = defaultdict(list)
    # for line in lines:
    #     section_name, doc_name = line.strip().split("/")
    #     sampled_doc_dict[section_name].append(doc_name)

    # brat_output_dir = prepare_brat(config, sampled_doc_dict)
    # copy_brat_configs(config.brat_config.base_dir, brat_output_dir)
    # logger.info("Data for BRAT labeling is created to: %s", brat_output_dir)

    # Resolve brat ann data
    resolve_brat(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
