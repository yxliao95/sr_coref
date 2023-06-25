import os
from collections import defaultdict

import hydra

module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


def get_previous_output_base_dir(config):
    current_iter = int(config.current_iter)
    if current_iter > 0:
        return os.path.join(os.path.split(config.output.base_dir)[0], f"iter_{current_iter-1}")
    else:
        return None


def get_previous_model_training_data_base_dir(config):
    previous_output_base_dir = get_previous_output_base_dir(config)
    if previous_output_base_dir is None:
        return None
    base_name = os.path.basename(config.output.model_training_data.base_dir)
    previous_model_training_data_base_dir = os.path.join(previous_output_base_dir, base_name)
    return previous_model_training_data_base_dir


def get_previous_labeled_pool_dict(config):
    """Read the `labeled_pool_info.txt` from the previous iter dir"""
    previous_output_base_dir = get_previous_output_base_dir(config)
    if previous_output_base_dir is None:
        return None
    previous_labeled_pool_info_file = os.path.join(
        previous_output_base_dir, os.path.basename(config.output.log.labeled_pool_info_file)
    )
    if os.path.exists(previous_labeled_pool_info_file):
        with open(previous_labeled_pool_info_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sampled_doc_dict = defaultdict(list)
        for line in lines:
            if line.strip() == "":
                continue
            section_name, doc_name = line.strip().split("/")
            sampled_doc_dict[section_name].append(doc_name)
        return sampled_doc_dict
    else:
        return None


def get_current_labeled_pool_dict(config):
    """Read the `labeled_pool_info.txt` from the current iter dir"""
    with open(config.output.log.labeled_pool_info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sampled_doc_dict = defaultdict(list)
    for line in lines:
        section_name, doc_name = line.strip().split("/")
        sampled_doc_dict[section_name].append(doc_name)
    return sampled_doc_dict


def get_trainset_size(config):
    with open(config.output.log.labeled_pool_info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return len([i for i in lines if i.strip() != ""])


def remove_labeled_data_from_sampling_dict(previous_sampled_doc_dict, extra_info_dict):
    """_summary_

    Args:
        previous_sampled_doc_dict (_type_): The labeled doc that we want to prevent from
            sampling at current iteration.
        extra_info_dict (_type_): This dict has all the sampling information. We just need
            to remove the labeled doc from this dict.
    """
    for section_name, doc_list in extra_info_dict.items():
        to_be_delete = []
        for doc_name in doc_list.keys():
            if doc_name in previous_sampled_doc_dict[section_name]:
                to_be_delete.append(doc_name)
        for doc_name in to_be_delete:
            del extra_info_dict[section_name][doc_name]


@hydra.main(version_base=None, config_path=config_path, config_name="active_learning")
def main(config):
    print(get_previous_labeled_pool_dict(config))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
