import os

import torch
import torch.nn as nn
import wandb
from dataset import get_dataloader
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from utils import InfIterator, get_decay_parameter_names
import numpy as np
from peft import LoraConfig, get_peft_model

import pandas as pd
from fastchat.model import get_conversation_template
import torch.multiprocessing as mp
from copy import deepcopy
import time
import json
import random
import torch.nn.functional as F
from vllm import LLM, SamplingParams
from transformers import GenerationConfig


task_names_and_extractor_text = {
    # "multistep_arithmetic_two": "Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive or negative integer) is $",
    "tracking_shuffled_objects_five_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    "object_counting": "Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive integer) is $",
    # "date_understanding": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    # "disambiguation_qa": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C') is $",
    # "formal_fallacies": "Therefore, the final answer (use exact format: '$ valid' or '$ invalid') is $ ",
    # "geometric_shapes": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J') is $",
    # "salient_translation_error_detection": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F') is $",
    # "penguins_in_a_table": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    "causal_judgement": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "logical_deduction_five_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    "movie_recommendation": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $",
    # "navigate": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "web_of_lies": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "sports_understanding": "Therefore, the final answer (use exact format: '$ yes' or '$ no') is $ ",
    # "reasoning_about_colored_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J' or '\$K' or '\$L' or '\$M' or '\$N' or '\$O' or '\$P' or '\$Q' or '\$R') is $",
    "hyperbaton": "Therefore, the final answer (use exact format: '\$A' or '\$B') is $",
    # "ruin_names": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $",
    # "snarks": "Therefore, the final answer (use exact format: '\$A' or '\$B') is $",
    # "temporal_sequences": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $",
    # "boolean_expressions": "Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
}


def remove_parentheses_if_single_char(input_string):
    if input_string.startswith('(') and input_string.endswith(')') and len(input_string) == 3:
        return input_string[1:-1]
    return input_string


def get_goals_and_targets(data_dir, extractor_text, conversation_templates, n_train_data, n_test_data):
    addition = "."
    addition3 = ""
    train_goals = []
    train_targets = []
    test_goals = []
    test_targets = []
    # additionally added to facilitate final loss
    train_final_targets = []
    test_final_targets = []

    addition2 = extractor_text
    offset = 0

    if data_dir:
        if data_dir.endswith('.tsv'):
            train_data = pd.read_csv(data_dir, sep='\t', dtype=str)
        else:
            train_data = pd.read_csv(data_dir, dtype=str)
        train_targets = train_data['final_target'].astype(str).tolist()[offset:offset + n_train_data]
        if len(addition2) > 0:
            train_targets = [addition2 + remove_parentheses_if_single_char(target) for target in train_targets]

        if 'goal' in train_data.columns:
            train_goals = train_data['goal'].astype(str).tolist()[offset:offset + n_train_data]
            if len(addition) > 0:
                train_goals = [goal + addition for goal in train_goals]

        else:
            train_goals = [""] * len(train_targets)

        # Most datasets won't have it. Just the ones we are curating for this feature
        if 'final_target' in train_data.columns:
            train_final_targets = train_data['final_target'].astype(str).tolist()[offset:offset + n_train_data]
            if len(addition3) >= 0 and "llama-3" in conversation_templates:
                train_final_targets = [addition3 + remove_parentheses_if_single_char(target) for target in train_final_targets]
            elif "llama-2" in conversation_templates or "gemma-2" in conversation_templates:
                train_final_targets = [remove_parentheses_if_single_char(target) for target in train_final_targets]
        else:
            train_final_targets = [""] * len(train_targets)


        if data_dir and n_test_data > 0:
            if data_dir.endswith('.tsv'):
                test_data = pd.read_csv(data_dir, sep='\t', dtype=str)
            else:
                test_data = pd.read_csv(data_dir)
            test_targets = test_data['final_target'].astype(str).tolist()[
                           offset + n_train_data:offset + n_train_data + n_test_data]
            if 'goal' in test_data.columns:
                test_goals = test_data['goal'].astype(str).tolist()[offset + n_train_data:offset + n_train_data + n_test_data]
                if len(addition2) > 0:
                    test_targets = [addition2 + remove_parentheses_if_single_char(target) for target in test_targets]
                if len(addition) > 0:
                    test_goals = [goal + addition for goal in test_goals]
            else:
                test_goals = [""] * len(test_targets)

            # Again, Most datasets won't have it. Just the ones we are curating for this feature
            if 'final_target' in test_data.columns:
                test_final_targets = test_data['final_target'].astype(str).tolist()[offset + n_train_data:offset + n_train_data + n_test_data]
                if len(addition3) >= 0 and "llama-3" in conversation_templates:
                    test_final_targets = [addition3 + remove_parentheses_if_single_char(target) for target in test_final_targets]
                elif "llama-2" in conversation_templates or "gemma-2" in conversation_templates:
                    test_final_targets = [remove_parentheses_if_single_char(target) for target in test_final_targets]
            else:
                test_final_targets = [""] * len(test_targets)

        elif n_test_data > 0:
            test_targets = train_data['target'].astype(str).tolist()[
                           offset + n_train_data:offset + n_train_data + n_test_data]
            if 'goal' in train_data.columns:
                test_goals = train_data['goal'].astype(str).tolist()[
                             offset + n_train_data:offset + n_train_data + n_test_data]
                if len(addition) > 0:
                    test_goals = [goal + addition for goal in test_goals]
            else:
                test_goals = [""] * len(test_targets)

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    # print('Loaded {} train goals'.format(len(train_goals)))
    # print('Loaded {} test goals'.format(len(test_goals)))

    return train_goals, train_targets, test_goals, test_targets, train_final_targets, test_final_targets


def get_conv_template(conversation_templates):
    conv = get_conversation_template(conversation_templates)

    if 'gemma-2' in conversation_templates:
        conv.name = 'gemma-2'
    elif 'gemma' in conversation_templates:
        conv.name = 'gemma'
    elif 'llama-3' in conversation_templates:
        conv.name = 'llama-3'
    elif 'gpt2' in conversation_templates:
        conv.name = 'gpt2'
    else:
        raise NotImplementedError(f"Conversation template {conversation_templates} not implemented")

    if conv.name == 'zero_shot':
        conv.roles = tuple(['### ' + r for r in conv.roles])
        conv.sep = '\n'
    elif conv.name == 'llama-2':
        conv.system = "<s>[INST] "  # forcing to use no system instruction
        conv.sep2 = conv.sep2.strip()
    elif conv.name == 'llama-3':
        conv.system = " "  # not used in the system
    elif conv.name == 'gpt2':
        conv.system = " "  # not used in the system
    elif conv.name == 'gemma':
        # conv.system = "<bos><start_of_turn>"
        conv.system = "<bos>"
        conv.roles = ('user\n', 'model\n')
        # Handle rest manually inside implementation to avoid any potential issues
    return conv


def apply_conv_template(conv_template, goal, control, control_pos='post'):
    
    if conv_template.name == 'llama-2':
        raise NotImplementedError("Llama-2 is not supported")
    elif conv_template.name == 'llama-3':
        conv_template.messages = []
        full_input = ""

        # user role slice
        full_input += "<|start_header_id|>user<|end_header_id|>\n\n"  # are u sure?

        if control_pos == "post":
            separator = " "
            full_input += goal
            if control.startswith(" "):
                control = control[1:]
            full_input = full_input + " " + control
        elif control_pos == "pre":
            full_input += control
            full_input += " "
            full_input += goal
        
        # assistant role slice
        full_input += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif conv_template.name == 'gemma-2':
        conv_template.messages = []
        full_input = ""

        # user role slice
        full_input += "<bos><start_of_turn>user\n"

        if control_pos == "post":
            separator = " "
            # goal_slice
            full_input += goal

            # control slice
            if control.startswith(" "):
                control = control[1:]
            full_input = full_input + " " + control
        elif control_pos == "pre":
            raise NotImplementedError # Not necessary to be implemented in our protocol

        # assistant role slice
        full_input += "<end_of_turn>\n<start_of_turn>model\n"

    elif conv_template.name == 'gpt2':
        conv_template.messages = []
        instruction = goal + " " + control
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt_template.format(instruction=instruction.rstrip())
    else:
        raise NotImplementedError(f"Conversation template {conv_template.name} not implemented")
    return full_input


def apply_template(conv_template, goal_control, reasoning, target):
    
    if conv_template.name == 'llama-2':
        raise NotImplementedError("Llama-2 is not supported")
    elif conv_template.name == 'llama-3':
        conv_template.messages = []
        full_input = ""
        full_input += goal_control
        full_input += reasoning
        full_input += " " ## added on Sept 15, 2024
        full_input += target
    elif conv_template.name == 'gemma-2':
        full_input = ""
        full_input += goal_control
        full_input += reasoning
        full_input += target
    elif conv_template.name == 'gpt2':
        conv_template.messages = []
        instruction = goal_control
        prompt_template = "{instruction}{reasoning} {target}"
        return prompt_template.format(instruction=instruction.rstrip(), reasoning=reasoning.rstrip(), target=target.rstrip())
    else:
        raise NotImplementedError(f"Conversation template {conv_template.name} not implemented")

    return full_input


class BBHEvalVLLMTrainer(object):
    def __init__(self, args) -> None:
        self.args = args
        if not hasattr(args, "model_paths") and hasattr(args, "eval_model_paths"):
            args.model_paths = args.eval_model_paths
        
        gen_config = GenerationConfig.from_pretrained(args.model_paths)
        print(gen_config)

        # VLLM setup
        self.llm = LLM(
            model=args.model_paths,
            tokenizer=args.model_paths,
            # max_model_len=1024,
            dtype="float16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
        )
        # temp = gen_config.temperature
        temp = 0.0
        self.params = SamplingParams(temperature=temp, top_p=gen_config.top_p, max_tokens=1024)
        # self.params4logprobs = SamplingParams(temperature=gen_config.temperature, top_p=gen_config.top_p, max_tokens=1, prompt_logprobs=1)
        self.params4logprobs = SamplingParams(temperature=temp, top_p=gen_config.top_p, max_tokens=1)

        self.output_dir = os.path.join(self.args.save_dir, self.args.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # get BBH data - filter by task_name if specified
        self.task_data = {}
        tasks_to_load = task_names_and_extractor_text
        if hasattr(args, 'task_name') and args.task_name in task_names_and_extractor_text:
            tasks_to_load = {args.task_name: task_names_and_extractor_text[args.task_name]}
            print(f"[BBH Eval] Running single task: {args.task_name}")
        else:
            print(f"[BBH Eval] Running all {len(tasks_to_load)} tasks")
        
        for task_name, extractor_text in tasks_to_load.items():
            data_dir = f'data/GreaTer_data/BBH/{task_name}.json'
            conversation_templates = args.eval_model_conversation_templates
            n_train_data, n_test_data = 50, 100
            train_goals, train_targets, test_goals, test_targets, train_final_target, test_final_target = get_goals_and_targets(data_dir, extractor_text, conversation_templates, n_train_data, n_test_data)
            self.task_data[task_name] = {
                "train_goals": train_goals,
                "train_targets": train_targets,
                "test_goals": test_goals,
                "test_targets": test_targets,
                "train_final_target": train_final_target,
                "test_final_target": test_final_target}
            print(f"  Loaded {task_name}: {len(test_goals)} test samples")
            
        
    def train(self):
        """
        Reproduce GreaTer paper (ICLR 2025) results.
        Evaluate with GREATER optimal prompt (Opt) and TextGrad prompt (TG).
        Compare against paper-reported accuracy.
        """

        # Paper-reported accuracy (from GreaTer Table)
        paper_acc = {
            "movie_recommendation": 0.57,
            "object_counting": 0.90,
            "tracking_shuffled_objects_five_objects": 0.70,
            "hyperbaton": 0.84,
            "causal_judgement": 0.57,
        }

        # Prompts to evaluate: Opt (GREATER) and TG (TextGrad)
        eval_prompts = {
            "movie_recommendation": {
                "Opt": "Use movie ratings data available here above movies for reference. ThisHOFF has an interesting analysis based solely options to options based movies ratings expect from the other movies you are asked ones mentioned here you.",
            },
            "object_counting": {
                "Opt": "Use only addition. Add think step by step. Finally give the actual correct answer.",
            },
            "tracking_shuffled_objects_five_objects": {
                "Opt": "Use this process as an explanation stepwise for each step until you get to as given above Alice has got originaly the following as follows.",
            },
            "hyperbaton": {
                "Opt": "Use the reasoning and examples you would step. Finally give the actual correct answer.",
            },
            "causal_judgement": {
                "Opt": "Use causal diagram. The correct option ask about whether there the variable C of about whether a specific cause is sufficient. The answer a causal relationship between C to D if the probability P that C occurs given E changes.",
            },
        }

        all_results = {}

        for task_name, task_data_ in self.task_data.items():
            if task_name not in eval_prompts:
                print(f"[SKIP] No prompts for task: {task_name}")
                continue

            task_results = {"paper_acc": paper_acc.get(task_name, None)}
            task_prompt_dict = eval_prompts[task_name]

            for method_name, control in task_prompt_dict.items():
                stpwatch_strt = time.time()
                print(f"\n{'='*60}")
                print(f"Task: {task_name} | Method: {method_name}")
                print(f"Prompt: {control[:80]}{'...' if len(control) > 80 else ''}")
                print(f"{'='*60}")

                test_goals = task_data_["test_goals"]
                test_targets = task_data_["test_targets"]
                test_final_target = task_data_["test_final_target"]
                goals, targets, final_targets = test_goals, test_targets, test_final_target

                # Step 1: Generate reasoning (CoT)
                conv_template = get_conv_template(self.args.eval_model_conversation_templates)
                goals_controls = [apply_conv_template(conv_template, goal, control) for goal in goals]
                outputs = self.llm.generate(goals_controls, self.params, use_tqdm=False)
                reasonings = [output.outputs[0].text for output in outputs]

                # Step 2: Extract final answer
                goals_reasonings_targets = [
                    apply_template(conv_template, goal_control, reasoning, task_names_and_extractor_text[task_name])
                    for goal_control, reasoning in zip(goals_controls, reasonings)
                ]
                outputs = self.llm.generate(goals_reasonings_targets, self.params4logprobs, use_tqdm=False)
                final_answers = [output.outputs[0].text.strip() for output in outputs]

                # Step 3: Calculate accuracy
                acc = sum(1 for fa, ft in zip(final_answers, final_targets) if fa == ft)
                acc = float(acc) / len(final_targets)

                inference_time = time.time() - stpwatch_strt
                print(f"  -> Accuracy: {acc:.4f} ({int(acc * len(final_targets))}/{len(final_targets)}) | Time: {inference_time:.1f}s")

                task_results[method_name] = {
                    "prompt": control,
                    "accuracy": acc,
                    "inference_time": inference_time,
                    "correct": int(acc * len(final_targets)),
                    "total": len(final_targets),
                }

            all_results[task_name] = task_results

            with open(os.path.join(self.output_dir, f'paper_prompts_{task_name}.json'), 'w') as f:
                json.dump(task_results, f, indent=4)

        # ============================================================
        # Print summary table (matching image format)
        # ============================================================
        print("\n\n" + "=" * 90)
        print("Logit Argmax Evaluation (100 samples, vLLM)")
        print("=" * 90)
        header = f"{'Task':<40} {'Paper':>8} {'vLLM-Opt':>10} {'vLLM-TG':>10}"
        print(header)
        print("-" * 90)
        avg_paper, avg_opt, avg_tg = [], [], []
        for task_name, task_results in all_results.items():
            p = task_results.get("paper_acc", 0)
            opt = task_results.get("Opt", {}).get("accuracy", 0)
            tg = task_results.get("TG", {}).get("accuracy", 0)
            avg_paper.append(p)
            avg_opt.append(opt)
            avg_tg.append(tg)
            short_name = task_name.replace("tracking_shuffled_objects_five_objects", "tracking_shuffled_objects")
            print(f"{short_name:<40} {p:>8.2f} {opt:>10.2f} {tg:>10.2f}")
        print("-" * 90)
        print(f"{'AVERAGE':<40} {sum(avg_paper)/len(avg_paper):>8.3f} {sum(avg_opt)/len(avg_opt):>10.3f} {sum(avg_tg)/len(avg_tg):>10.3f}")
        print("=" * 90)

        with open(os.path.join(self.output_dir, 'paper_prompts_all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

