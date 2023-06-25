import json
import logging
import os
import time
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

import active_learning.query_strategies as query_strategies
import nlp_ensemble.nlp_menbers.play_fastcoref as play_fastcoref

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


def model_inference(config, target_sections, log_dict=None):
    """The model inference results will be saved in `config.output.model_inference_csv_dir`.

    Args:
        config (_type_): _description_
        target_sections (list): Should be ["findings", "impression"].
        log_dict (dict): For logging

    Returns:
        extra_info_dict: provide mention scores of the processed documents. It looks like
            {'findings': {'s51816597.csv': {'mention_logits': tensor([-25.1743, -2...-17.2089])}}}
    """
    # Init fast-coref-joint
    model, subword_tokenizer, max_segment_len = play_fastcoref.init_coref_model(
        None, model_dir=config.coref_model.model_dir, doc_encoder_dir=config.coref_model.doc_encoder_dir
    )

    # Model prediction
    extra_info_dict = {}  # {'findings': {'s51816597.csv': {'mention_logits': tensor([-25.1743, -2...-17.2089])}}}
    # The model inference results will be saved in `/model_inference_csv`.
    processed_record_num_per_section = play_fastcoref.run(
        config,
        target_sections,
        model,
        subword_tokenizer,
        max_segment_len,
        section_docid_extra_output_dict=extra_info_dict,
    )

    if log_dict is not None:
        log_dict["processed_record_num"] = processed_record_num_per_section

    return extra_info_dict


def sampling_topk_doc_by_MDE(extra_info_dict: dict, sampling_nums: dict, log_dict=None):
    """Sampling topk doc by Mention Detection Entropy (the highest mention entropy of each document)

    Args:
        extra_info_dict (dict): from model inference
        sampling_nums (dict): _description_
        log_dict (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: sampled_doc_dict. {"section_name": doc_list}
    """
    sampled_doc_dict = {}

    # Sampling docs based on entropy
    for section_name, doc_extra_output_dict in extra_info_dict.items():
        doc_csv_name_list = []
        doc_highest_entropy_list = []
        # Compute mention entropy for each doc and select the highest mention entropy
        for doc_csv_name, extra_output_dict in doc_extra_output_dict.items():
            mention_logits_tensor = extra_output_dict["mention_logits"]
            mention_entropy_tensor = query_strategies.mention_detection_entropy(mention_logits_tensor)
            highest_entropy_scaler = torch.max(mention_entropy_tensor)
            doc_csv_name_list.append(doc_csv_name)
            doc_highest_entropy_list.append(highest_entropy_scaler.unsqueeze(dim=0))
        doc_highest_entropy_tensor = torch.cat(doc_highest_entropy_list, dim=0)
        doc_highest_entropy_tensor_nanTo0 = torch.nan_to_num(doc_highest_entropy_tensor)
        topk_entropy_tensor, topk_indices_tensor = doc_highest_entropy_tensor_nanTo0.topk(sampling_nums[section_name])

        topk_indices_array = topk_indices_tensor.detach().cpu().numpy()
        doc_csv_name_array = np.array(doc_csv_name_list)
        sampled_doc_array = doc_csv_name_array[topk_indices_array]

        if log_dict is not None:
            sampled_doc_dict[section_name] = sampled_doc_array.tolist()
            log_dict[section_name]["docs"] = sampled_doc_array.tolist()
            log_dict[section_name]["entropy"] = topk_entropy_tensor.tolist()

    return sampled_doc_dict


def log_runtime_info(config, log_dict, startTime=None):
    # Log runtime information
    OmegaConf.save(config=config, f=os.path.join(config.output.log.config_file), resolve=True)
    with open(config.output.log.sampling_file, "w", encoding="UTF-8") as f:
        log_out = {
            "Iteration number": config.current_iter,
            "Sampling number": config.sampling_num,
            "Time cost": f"{time.time() - startTime:.2f}s" if startTime is not None else "N/A",
            "Number of processed docs": log_dict["processed_record_num"],
            "Sampled document details": {
                "findings": {
                    "doc_num": len(log_dict["findings"]["docs"]),
                    "docs": log_dict["findings"]["docs"],
                    "highest mention entropy": log_dict["findings"]["entropy"],
                },
                "impression": {
                    "doc_num": len(log_dict["impression"]["docs"]),
                    "docs": log_dict["impression"]["docs"],
                    "highest mention entropy": log_dict["impression"]["entropy"],
                },
            },
        }
        f.write(json.dumps(log_out, indent=2))
        f.write("\n\n")
    return log_out


@hydra.main(version_base=None, config_path=config_path, config_name="active_learning")
def main(config):
    target_sections = ["findings", "impression"]
    sampling_nums = {
        "findings": config.sampling_num // 2,
        "impression": config.sampling_num - (config.sampling_num // 2),
    }
    startTime = time.time()
    log_dict = defaultdict(dict)

    extra_info_dict = model_inference(config, target_sections, log_dict)

    sampled_doc_dict = sampling_topk_doc_by_MDE(extra_info_dict, sampling_nums, log_dict)

    # Save labeled pool info
    with open(config.output.log.labeled_pool_info_file, "w", encoding="utf-8") as f:
        for section_name, doc_list in sampled_doc_dict.items():
            f.write("\n".join([f"{section_name}/{doc_name}" for doc_name in doc_list]))
            f.write("\n")

    log_runtime_info(config, log_dict, startTime)


if __name__ == "__main__":
    # sys.argv.append("nlp_ensemble@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
