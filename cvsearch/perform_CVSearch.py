import sys
import os
import argparse
import json
import warnings
import torch
import spacy
import numpy as np
from tqdm import tqdm
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
from models.modeling_llava import ModelGlobalLocal, ModelLocal
from models.modeling_internvl import ModelInternvl
from models.modeling_qwenvl import ModelQwenVL
from models.modeling_sam3 import sam3_inference
from CVSearch import get_cvsearch_response, get_direct_response

warnings.filterwarnings("ignore")
_original_np_load = np.load

def get_chunk(lst, n, k):
    subarrays = [[] for _ in range(n)]
    for i in range(n):
        subarrays[i] = lst[i::n]
    return subarrays[k]

def get_basename(path):
    return os.path.basename(path)

def _patched_np_load(*args, **kwargs):
    kwargs['allow_pickle'] = True
    return _original_np_load(*args, **kwargs)

np.load = _patched_np_load

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--split-num", type=int, default=4)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--nlp-model-path", type=str, default="models/en_core_web_sm-3.8.0")
    parser.add_argument("--sam-model-path", type=str, default="models/facebook/sam3/sam3.pt")
    parser.add_argument("--model-path", type=str, default="models/Qwen2.5-VL-7B")
    parser.add_argument("--annotation_path", type=str, default="datasets/hr_data")
    parser.add_argument("--benchmark", type=str,
                        choices=["vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld-lite", "treebench",
                                 "fines-bench_option", "fines-bench_reasoning"], default="hr-bench_4k")
    parser.add_argument("--direct-answer", action="store_true")
    args = parser.parse_args()

    model_path = os.path.join(args.root_path, args.model_path)
    annoataion_path = os.path.join(args.root_path, args.annotation_path)
    benchmark = args.benchmark
    split_num = args.split_num
    if args.answers_file is None:
        answers_dir = f"CVsearch/eval/answers/{benchmark}"
        answers_dir = os.path.join(answers_dir, os.path.basename(args.model_path))
        os.makedirs(answers_dir, exist_ok=True)
        answer_tag = 'cvsearch' if not args.direct_answer else "direct_answer"
        args.answers_file = os.path.join(answers_dir, f"{answer_tag}.jsonl")
        print(args.answers_file)

    config_path = os.path.join(model_path, "config.json")
    config = json.load(open(config_path, "r"))

    if "llava" in model_path.lower():
        if "anyres" in config['image_aspect_ratio']:
            search_model = ModelGlobalLocal(model_path=model_path, conv_type="qwen_1_5", patch_scale=1.2,bias_value=0.6)
        else:
            search_model = ModelLocal(model_path=model_path, conv_type="v1", patch_scale=None, bias_value=0.2)

        def pop_limit_func(max_depth):
            return max_depth * 3

        search_kwargs = {
            "pop_limit": pop_limit_func,
            "threshold_descrease": [0.1, 0.1, 0.2],
            "answering_confidence_threshold_lower": 0,
            "answering_confidence_threshold_upper": 0.6,
            "fast_threshold": 0.8
        }
    elif "internvl" in model_path.lower():
        search_model = ModelInternvl(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16,patch_scale=1.2)

        def pop_limit_func(max_depth):
            return max_depth * 3

        search_kwargs = {
            "pop_limit": pop_limit_func,
            "threshold_descrease": [0.1, 0.1, 0.2],
            "answering_confidence_threshold_lower": -0.2,
            "answering_confidence_threshold_upper": 0.2,
            "fast_threshold": 0.6
        }
    elif "qwen" in model_path.lower():
        if "32b" in model_path.lower():
            kwargs = {"load_in_8bit": True}
        else:
            kwargs = {}
        search_model = ModelQwenVL(model_path=model_path, device="cuda:0", torch_dtype=torch.bfloat16,patch_scale=1.2, **kwargs)

        def pop_limit_func(max_depth):
            return max_depth * 3

        search_kwargs = {
            "pop_limit": pop_limit_func,
            "threshold_descrease": [0.05, 0.1, 0.2],
            "answering_confidence_threshold_lower": 0,
            "answering_confidence_threshold_upper": 0.9,
            "fast_threshold": 0.6 #0.8
        }
    else:
        raise ValueError(f"Model {model_path} not supported")

    print(search_kwargs)

    nlp = spacy.load(name=os.path.join(args.root_path, args.nlp_model_path))
    sam3 = sam3_inference(model_path=os.path.join(args.root_path, args.sam_model_path))
    decomposed_question_template = "What is the appearance of the {}?"

    ic_examples_path = f"ic_examples/{benchmark}.json"
    if benchmark == "vstar" and "llava" in model_path.lower():
        m = json.load(
            open(os.path.join(annoataion_path, f"{benchmark}/annotation_{benchmark}_updated.json"), "r"))  # _updated
    else:
        if benchmark == "fines-bench_option" or benchmark == "fines-bench_reasoning":
            m = json.load(open(os.path.join(annoataion_path, f"fines-bench/annotation_{benchmark}.json"), "r"))
            benchmark = "fines-bench"
        else:
            m = json.load(open(os.path.join(annoataion_path, f"{benchmark}/annotation_{benchmark}.json"), "r"))
    m = get_chunk(m, args.num_chunks, args.chunk_idx)

    num = 1
    results_file = open(args.answers_file, 'w')
    for annotation in tqdm(m):
        print(f"Sample Num: {num}")
        if not args.direct_answer:
            response = get_cvsearch_response(
                sam_model=sam3,
                zoom_model=search_model,
                nlp_model=nlp,
                annotation=annotation,
                ic_examples=json.load(open(ic_examples_path, "r")),
                decomposed_question_template=decomposed_question_template,
                image_folder=os.path.join(annoataion_path, f"{benchmark}"),
                **search_kwargs,
            )
        else:
            response = get_direct_response(
                zoom_model=search_model,
                annotation=annotation,
                image_folder=os.path.join(annoataion_path, f"{benchmark}"),
            )
        annotation['output'] = response
        results_file.write(json.dumps(annotation) + "\n")
        num += 1
    results_file.close()
