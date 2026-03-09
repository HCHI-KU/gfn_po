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
    # "tracking_shuffled_objects_five_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    "object_counting": "Therefore, the final answer (use exactly this format: \$NUMBER\$, where NUMBER is a positive integer) is $",
    # "date_understanding": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    # "disambiguation_qa": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C') is $",
    # "formal_fallacies": "Therefore, the final answer (use exact format: '$ valid' or '$ invalid') is $ ",
    # "geometric_shapes": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J') is $",
    # "salient_translation_error_detection": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F') is $",
    # "penguins_in_a_table": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    # "causal_judgement": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "logical_deduction_five_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E') is $",
    # "movie_recommendation": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D') is $",
    # "navigate": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "web_of_lies": "Therefore, the final answer (use exact format: '$ Yes' or '$ No') is $ ",
    # "sports_understanding": "Therefore, the final answer (use exact format: '$ yes' or '$ no') is $ ",
    # "reasoning_about_colored_objects": "Therefore, the final answer (use exact format: '\$A' or '\$B' or '\$C' or '\$D' or '\$E' or '\$F' or '\$G' or '\$H' or '\$I' or '\$J' or '\$K' or '\$L' or '\$M' or '\$N' or '\$O' or '\$P' or '\$Q' or '\$R') is $",
    # "hyperbaton": "Therefore, the final answer (use exact format: '\$A' or '\$B') is $",
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
                    train_final_targets = [remove_parentheses_if_single_char(target) for target in train_final_targets]
            else:
                test_final_targets = [""] * len(train_targets)

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
        
        # get BBH data
        self.task_data = {}
        for task_name, extractor_text in task_names_and_extractor_text.items():
            data_dir = f'data/GreaTer_data/BBH/{task_name}.json'
            conversation_templates = args.conversation_templates
            n_train_data, n_test_data = 50, 50
            train_goals, train_targets, test_goals, test_targets, train_final_target, test_final_target = get_goals_and_targets(data_dir, extractor_text, conversation_templates, n_train_data, n_test_data)
            self.task_data[task_name] = {
                "train_goals": train_goals,
                "train_targets": train_targets,
                "test_goals": test_goals,
                "test_targets": test_targets,
                "train_final_target": train_final_target,
                "test_final_target": test_final_target}
            
        
    def train(self):

        control_dict = {
            'object_counting_from_greater': [
                " The first, the first, the first, the first, the first, the first, the first",
                "Let’s think step by step.",
                "Break it down, step by step.",
                "You will answer a reasoning question about counting objects. Think step by step, considering the context of the question and using it to inform your answer. Be explicit in your counting process, breaking it down...", # 
                "Let’s think step by step.",
                "Use only addition. Add think step by step. Finally give the actual correct answer."
        ]}

        # control_dict = {
        #     'formal_fallacies_from_greater': [
        #         "Determine the validity of the given argument.",
        #         "Simplify and analyze.",
        #         "You will answer a reasoning question by explicitly identifying the key relationships between the premises and the conclusion, and explaining how they lead to the conclusion. Use clear and concise language to facilitate understanding, and...",
        #         "Analyze the argument step by step, considering premises, logical connections, and conditional statements. Identify the conclusion and evaluate its validity, considering sufficient and necessary conditions, counterexamples, and alternative scenarios.",
        #         "Use formal notation and and think step by step. Finally give the actual correct answer."
        # ]}

        # control_dict = {
        #     'normal': [
        #         "",
        #         "Let's think step by step.",
        #         "Use proper logical reasoning and think step by step. Finally give the actual correct answer.",
        #         "Let's work this out in a step by step way to be sure we have the right answer.",
        # ]}
        
        # add controls from OPRO prompt history
        with open(self.args.prompt_history_file, "r") as f:
            prompt_history = json.load(f)
        
        for pr in prompt_history[0].keys():
            # control_dict[pr] = prompt_history[0][pr]
            random.seed(42)
            control_dict[pr] = random.sample(prompt_history[0][pr], 100)
        
        for task_name, task_data_ in self.task_data.items():
            
            for control_task_name in control_dict.keys():
                controls = control_dict[control_task_name]
                control_task_log = {"control_task_name": control_task_name, "controls": []}

                for control in controls:
                    stpwatch_strt = time.time()
                    print(f"Task: {task_name}, Control_task_name: {control_task_name}, Control: {control}")

                    # '''
                    train_goals = task_data_["train_goals"]
                    train_targets = task_data_["train_targets"]
                    train_final_target = task_data_["train_final_target"]
                    test_goals = task_data_["test_goals"]
                    test_targets = task_data_["test_targets"]
                    test_final_target = task_data_["test_final_target"]

                    goals, targets, final_targets = train_goals, train_targets, train_final_target
                    # goals, targets, final_targets = test_goals, test_targets, test_final_target
                
                    # apply chat template
                    conv_template = get_conv_template(self.args.conversation_templates)
                    goals_controls = [apply_conv_template(conv_template, goal, control) for goal in goals]
                    outputs = self.llm.generate(goals_controls, self.params, use_tqdm=False)
                    reasonings = [output.outputs[0].text for output in outputs]
                    # reasonings = ["" for o in train_goals_controls]
                    # print(train_goals_controls[0])
                    # print(reasonings[0])
                    # exit()
                    
                    goals_reasonings_targets = [apply_template(conv_template, goal_control, reasoning, task_names_and_extractor_text[task_name]) for goal_control, reasoning in zip(goals_controls, reasonings)]
                    # print(train_goals_reasonings_targets[0])

                    outputs = self.llm.generate(goals_reasonings_targets, self.params4logprobs, use_tqdm=False)
                    final_answers = [output.outputs[0].text.strip() for output in outputs]
                    # print(final_answers)
                    # print(train_final_target)
                    # exit()

                    acc = 0
                    for final_answer, final_target in zip(final_answers, final_targets):
                        if final_answer == final_target:
                            acc += 1
                    acc = float(acc) / len(final_targets)

                    inference_time = time.time() - stpwatch_strt
                    print(f"Accuracy: {acc} / Inference time: {inference_time}")
                    print("=" * 50)
                    control_task_log["controls"].append({"control": control, "accuracy": acc, "inference_time": inference_time})
                
            
                with open(os.path.join(self.output_dir, f'task_{task_name}_prompt_{control_task_name}.json'), 'w') as f:
                    json.dump(control_task_log, f, indent=4)