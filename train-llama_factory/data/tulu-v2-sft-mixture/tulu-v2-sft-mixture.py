# import json
# import datasets
#
#
# _DESCRIPTION = "tulu2 sft data(the sft dataset of open-instruct)."
#
# _CITATION = """\
# @misc{wang2023far,
#    title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources},
#    author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
#    year={2023},
#    eprint={2306.04751},
#    archivePrefix={arXiv},
#    primaryClass={cs.CL}
# }
# """
#
# _HOMEPAGE = "https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture"
# _LICENSE = "odc-by"
# _URL = ["https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture/resolve/main/data/train-00000-of-00003-99ee8754042a69f6.json",
# "
#
#
# class BelleMultiturn(datasets.GeneratorBasedBuilder):
#
#     VERSION = datasets.Version("0.0.0")
#
#     def _info(self):
#         features = datasets.Features({
#             "conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}]
#         })
#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=features,
#             homepage=_HOMEPAGE,
#             license=_LICENSE,
#             citation=_CITATION
#         )
#
#     def _split_generators(self, dl_manager: datasets.DownloadManager):
#         file_path = dl_manager.download(_URL)
#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={
#                     "filepath": file_path
#                 }
#             )
#         ]
#
#     def _generate_examples(self, filepath: str):
#         with open(filepath, "r", encoding="utf-8") as f:
#             for key, row in enumerate(f):
#                 data = json.loads(row)
#                 conversations = []
#                 prompt = data["instruction"].strip()
#                 response = data["output"].strip()
#
#                 assist_idx = prompt.rfind("Assistant:")
#                 human_idx = prompt.rfind("Human:")
#                 query = prompt[human_idx+6:assist_idx].strip()
#                 prompt = prompt[:human_idx].strip()
#                 conversations.insert(0, {"from": "gpt", "value": response})
#                 conversations.insert(0, {"from": "human", "value": query})
#
#                 while prompt.rfind("Assistant:") != -1:
#                     assist_idx = prompt.rfind("Assistant:")
#                     human_idx = prompt.rfind("Human:")
#                     if human_idx != -1:
#                         old_query = prompt[human_idx+6:assist_idx].strip()
#                         old_resp = prompt[assist_idx+10:].strip()
#                         conversations.insert(0, {"from": "gpt", "value": old_resp})
#                         conversations.insert(0, {"from": "human", "value": old_query})
#                     else:
#                         break
#                     prompt = prompt[:human_idx].strip()
#
#                 yield key, {"conversations": conversations}
