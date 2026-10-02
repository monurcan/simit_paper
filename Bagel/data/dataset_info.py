# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'data_root/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'data_root/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'data_root/bagel_example/editing/parquet_info/seedxedit_multi.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'data_root/bagel_example/vlm/images',
			'jsonl_path': 'data_root/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}


# SIMIT training sets register themselves through registry.json files written next
# to their jsonl by the data-preparation scripts (prepare_vqa_data_v2_dfadaptive.py,
# prepare_vqa_data_pool.py, register_real_rag_datasets.py), so no entries are
# hand-added here.
import glob as _glob
import json as _json
import os as _os

for _registry_path in sorted(_glob.glob(
    _os.path.join(_os.path.dirname(__file__), "..", "data_root", "*", "**", "registry.json"), recursive=True
)):
    with open(_registry_path) as _f:
        DATASET_INFO['vlm_sft'].update(_json.load(_f))
