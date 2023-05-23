import ast
import logging
import os
import shutil
from multiprocessing import Event

import hydra
import pandas as pd

# fast-coref module
from data_processing.utils import get_tokenizer

from active_learning.utils import get_previous_model_training_data_base_dir
from common_utils.common_utils import check_and_create_dirs, check_and_remove_dirs
from common_utils.coref_utils import ConllToken, resolve_mention_and_group_num
from common_utils.file_checker import FileChecker
from coreference_resolution.data_preprocessing.mimic_cxr_conll2jsonlines import (
    minimize_partition,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILE_CHECKER = FileChecker()
START_EVENT = Event()
logger = logging.getLogger()

module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

SEG_LEN = 4096


def build_individua_conll(config, keep_0_coref_docs=True):
    """manual labeled csv files -> individual conll files"""
    spacy_nametyle = config.name_style.spacy.column_name
    gt_namestyle = config.name_style.mimic_cxr_gt.column_name
    csv_input_base_dir = config.output.temp.manual_annotation_csv_dir
    conll_output_base_dir = config.output.temp.individual_conll_dir

    for section_name in ["findings", "impression"]:
        csv_input_dir = os.path.join(csv_input_base_dir, section_name)
        conll_output_dir = os.path.join(conll_output_base_dir, section_name)
        check_and_create_dirs(conll_output_dir)

        for doc_name in FILE_CHECKER.filter(os.listdir(csv_input_dir)):
            doc_id = doc_name.rstrip(".csv")
            output_file_path = os.path.join(conll_output_dir, f"{doc_id}.conll")

            BEGIN = f"#begin document ({doc_id}_{section_name}); part 0\n"
            SENTENCE_SEPARATOR = "\n"
            END = "#end document\n"

            # Resolve CSV file
            sentenc_list: list[list[ConllToken]] = []
            df = pd.read_csv(os.path.join(csv_input_dir, doc_name), index_col=0, na_filter=False)
            _, coref_group_num = resolve_mention_and_group_num(df, gt_namestyle.coref_group_conll)

            # Write .conll file only if doc has at least one coref group
            # New: choose to keep 0 coref docs based on `keep_0_coref_docs`
            if coref_group_num > 0 or keep_0_coref_docs:
                sentence_id = 0
                while True:
                    token_list: list[ConllToken] = []
                    df_sentence = df[df.loc[:, spacy_nametyle.sentence_group] == sentence_id].reset_index()
                    if df_sentence.empty:
                        break
                    for _idx, data in df_sentence.iterrows():
                        # Skip all whitespces like "\n", "\n " and " ".
                        if str(data[spacy_nametyle.token]).strip() == "":
                            continue
                        conllToken = ConllToken(
                            doc_id + "_" + section_name, sentence_id, _idx, data[spacy_nametyle.token]
                        )
                        coref_col_cell = data[gt_namestyle.coref_group_conll]
                        if isinstance(coref_col_cell, str) and coref_col_cell != "-1":
                            conllToken.add_coref_label("|".join(ast.literal_eval(coref_col_cell)))
                        token_list.append(conllToken)
                    sentenc_list.append(token_list)
                    sentence_id += 1

                with open(output_file_path, "w", encoding="UTF-8") as out:
                    out.write(BEGIN)
                    for sent in sentenc_list:
                        # Skip empty sentence
                        if len(sent) == 1 and sent[0].tokenStr == "":
                            continue
                        for tok in sent:
                            out.write(tok.get_conll_str() + "\n")
                        out.write(SENTENCE_SEPARATOR)
                    out.write(END)


def build_aggregrated_conll(config):
    conll_input_base_dir = config.output.temp.individual_conll_dir
    conll_output_dir = config.output.model_training_data.conll
    # The output conll file is written in "append" mode, so we need to delete and recreate it.
    check_and_remove_dirs(conll_output_dir)
    check_and_create_dirs(conll_output_dir)

    # Aggregrate conll one by one for train.conll
    output_conll_file = os.path.join(conll_output_dir, "train.conll")
    with open(output_conll_file, "a", encoding="UTF-8") as f_out:
        for section_name in ["findings", "impression"]:
            conll_input_dir = os.path.join(conll_input_base_dir, section_name)
            for doc_name in FILE_CHECKER.filter(os.listdir(conll_input_dir)):
                input_conll_file = os.path.join(conll_input_dir, doc_name)
                with open(input_conll_file, "r", encoding="UTF-8") as f_in:
                    f_out.write("".join(f_in.readlines()))
                    f_out.write("\n")

    # Copy the train/test.conll from resources
    dev_conll_file = config.reuse_conll.dev_file
    shutil.copyfile(dev_conll_file, os.path.join(conll_output_dir, "dev.conll"))
    test_conll_file = config.reuse_conll.test_file
    shutil.copyfile(test_conll_file, os.path.join(conll_output_dir, "test.conll"))


def build_jsonlines(config):
    tokenizer = get_tokenizer(config.coref_model.doc_encoder_dir)

    input_dir = config.output.model_training_data.conll
    output_dir = config.output.model_training_data.longformer
    check_and_remove_dirs(output_dir)
    check_and_create_dirs(output_dir)

    for doc_name in FILE_CHECKER.filter(os.listdir(input_dir)):
        split_name = doc_name.split(".")[0]
        output_path = os.path.join(output_dir, f"{split_name}.{SEG_LEN}.jsonlines")
        input_path = os.path.join(input_dir, doc_name)
        minimize_partition(input_path, output_path, tokenizer, SEG_LEN)


def concat_previous_conll_and_jsonlines(config):
    previous_model_training_data_base_dir = get_previous_model_training_data_base_dir(config)
    if previous_model_training_data_base_dir is None:
        return
    previous_train_conll_file = os.path.join(previous_model_training_data_base_dir, "conll", "train.conll")
    previous_train_jsonlines_file = os.path.join(
        previous_model_training_data_base_dir, "longformer", f"train.{SEG_LEN}.jsonlines"
    )

    curr_train_conll_file = os.path.join(config.output.model_training_data.conll, "train.conll")
    curr_train_jsonlines_file = os.path.join(config.output.model_training_data.longformer, f"train.{SEG_LEN}.jsonlines")

    # concat prev and curr conll files
    with open(previous_train_conll_file, "r", encoding="utf-8") as f:
        prev_train_conll_lines = f.readlines()
    with open(curr_train_conll_file, "r", encoding="utf-8") as f:
        curr_train_conll_lines = f.readlines()
    with open(curr_train_conll_file, "w", encoding="utf-8") as f:
        f.write("".join(prev_train_conll_lines))
        f.write("".join(curr_train_conll_lines))

    # concat prev and curr jsonlines files
    with open(previous_train_jsonlines_file, "r", encoding="utf-8") as f:
        prev_train_jsonlines = f.readlines()
    with open(curr_train_jsonlines_file, "r", encoding="utf-8") as f:
        curr_train_jsonlines = f.readlines()
    with open(curr_train_jsonlines_file, "w", encoding="utf-8") as f:
        f.write("".join(prev_train_jsonlines))
        f.write("".join(curr_train_jsonlines))


@hydra.main(version_base=None, config_path=config_path, config_name="active_learning")
def main(config):
    # build_individua_conll(config)
    # build_aggregrated_conll(config)
    build_jsonlines(config)


if __name__ == "__main__":
    # sys.argv.append("nlp_ensemble@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
