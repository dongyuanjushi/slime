from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from examples.cua_vlm_multi_turn.base_env import BaseInteractionEnv

from examples.cua_vlm_multi_turn.prompt import SYSTEM_PROMPT_QWEN_3_PLANNING
import re
import json
from typing import Any, Optional, Tuple, Dict, List
import json5
from PIL import Image
from io import BytesIO
import base64

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_cua"

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)

def adjust_coordinates(x: float, y: float, original_width: int = None, original_height: int = None):
    if not (original_width and original_height):
        return int(x), int(y)
    x_scale = original_width / 999
    y_scale = original_height / 999
    return int(x * x_scale), int(y * y_scale)


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)

def parse_action_and_pyautogui_code(raw_response: str) -> Tuple[str, Optional[dict]]:
    """
    Parse the LLM response to extract the action description and tool call.
    """
    action_description = raw_response
    tool_call_dict = None

    action_match = re.search(r'Action:\s*(.+?)(?=<tool_call>|$)', raw_response, re.DOTALL | re.IGNORECASE)
    if action_match:
        action_description = action_match.group(1).strip()

    tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', raw_response, re.DOTALL)
    if tool_call_match:
        tool_call_str = tool_call_match.group(1).strip()
        try:
            try:
                tool_call_dict = json5.loads(tool_call_str)
            except Exception:
                tool_call_dict = json.loads(tool_call_str)
        except Exception as e:
            tool_call_dict = None

    return action_description, tool_call_dict

def process_tool_call(tool_call: Dict[str, Any], original_width: int = None, original_height: int = None):
    """Convert tool call to pyautogui code string."""
    if tool_call is None:
        return ""

    pyautogui_code: List[str] = []

    try:
        action = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        if action == "left_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
            else:
                pyautogui_code.append("pyautogui.click()")

        elif action == "right_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.rightClick({adj_x}, {adj_y})")
            else:
                pyautogui_code.append("pyautogui.rightClick()")

        elif action == "double_click":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.doubleClick({adj_x}, {adj_y})")
            else:
                pyautogui_code.append("pyautogui.doubleClick()")

        elif action == "type":
            text = args.get("text", "")
            pyautogui_code.append(f"pyautogui.typewrite('{text}')")

        elif action == "key":
            keys = args.get("keys", [])
            if isinstance(keys, list):
                keys_str = ", ".join([f"'{key}'" for key in keys])
                if len(keys) > 1:
                    pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                else:
                    pyautogui_code.append(f"pyautogui.press({keys_str})")

        elif action == "scroll":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")
            pixels = args.get("pixels", 0)
            pyautogui_code.append(f"pyautogui.scroll({pixels})")
            
        elif action == "hscroll":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")
            pixels = args.get("pixels", 0)
            pyautogui_code.append(f"pyautogui.hscroll({pixels})")

        elif action == "wait":
            pyautogui_code.append("WAIT")

        elif action == "terminate":
            if "status" in args:
                if args["status"] == "success":
                    pyautogui_code.append("DONE")
                else:
                    pyautogui_code.append("FAIL")

        elif action == "mouse_move":
            if "coordinate" in args:
                x, y = args["coordinate"]
                adj_x, adj_y = adjust_coordinates(x, y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.moveTo({adj_x}, {adj_y})")

        elif action == "left_click_drag":
            if "start_coordinate" in args:
                start_x, start_y = args["start_coordinate"]
                adj_start_x, adj_start_y = adjust_coordinates(start_x, start_y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.moveTo({adj_start_x}, {adj_start_y})")
            if "end_coordinate" in args:
                end_x, end_y = args["end_coordinate"]
                adj_end_x, adj_end_y = adjust_coordinates(end_x, end_y, original_width, original_height)
                pyautogui_code.append(f"pyautogui.dragTo({adj_end_x}, {adj_end_y})")

    except (json.JSONDecodeError, KeyError) as e:
        return ""

    if len(pyautogui_code) > 0:
        if "DONE" in pyautogui_code or "WAIT" in pyautogui_code or "FAIL" in pyautogui_code:
            return pyautogui_code[0]
        else:
            pyautogui_code.insert(0, "import pyautogui")
            return "; ".join(pyautogui_code)

    return ""


def bytes_to_pil_image(image_data: bytes) -> Image.Image:
    return Image.open(BytesIO(image_data))


def base64_to_pil_image(base64_str: str) -> Image.Image:
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",", 1)[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """
    Encode a single observation turn that may include images/videos in the content list.
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer


async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
        _encode_observation_for_generation(
            tokenizer,
            processor,
            next_user_message,
            sample_metadata,
            args.apply_chat_template,
            args.apply_chat_template_kwargs,
        )
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> bool:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed: int):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


def _build_messages_and_images_for_processor(
    system_prompt: str,
    user_instruction: str,
    action_history: list[str],
    screenshot_history: list[str],
    last_k_screenshots: int = 3
) -> list[dict]:
    """
    Build the text messages for the processor with <image> placeholders.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
    ]
    
    assert len(screenshot_history) == len(action_history) + 1, "Screenshot history must be one longer than action history"
    
    images = []
    
    prev_start = 0
    
    if len(screenshot_history) > last_k_screenshots:
        prev_start = len(screenshot_history) - last_k_screenshots
    
    for i in range(prev_start, len(screenshot_history)):
        messages.append({"role": "user", "content": "<image>"})
        images.append(screenshot_history[i])
        if i < len(screenshot_history) - 1:
            messages.append({"role": "assistant", "content": action_history[i]})

    return messages, images

async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env, env_module, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()
    try:
        task_config = sample.metadata.get("config", {})
        initial_response = env.reset(task_config)
        system_prompt = SYSTEM_PROMPT_QWEN_3_PLANNING
        instruction = sample.metadata.get("instruction", "")
        initial_screenshot = initial_response.get("screenshot")
        vm_id = initial_response.get("vm_id")
        
        budget = None
        if args.rollout_max_context_len is not None:
            budget = args.rollout_max_context_len - len(sample.tokens)
        elif sampling_params.get("max_new_tokens") is not None:
            budget = sampling_params["max_new_tokens"] - len(sample.tokens)
        
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        cur_sampling_params = sampling_params
        action_history = []
        screenshot_history = []
        screenshot_history.append(initial_screenshot)
        
        for turn_idx in range(config["max_turns"]):
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget
                
            current_messages, current_images = _build_messages_and_images_for_processor(
                system_prompt=system_prompt,
                user_instruction=instruction,
                action_history=action_history,
                screenshot_history=screenshot_history
            )
            
            sample.tokens = state.tokenizer.apply_chat_template(current_messages, tokenize=True)

            response_text, new_response_tokens, new_response_log_probs, finish_type = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_images, state.tokenizer
            )
            
            action_description, tool_call = parse_action_and_pyautogui_code(response_text)
            action_code = process_tool_call(tool_call)
            step_response = env.step(action_code, vm_id)
            
            action_history.append(action_description)
            
            # _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            # budget = _update_budget(budget, len(new_response_tokens))
            
            is_finished = step_response.get("is_finished")
            reward = step_response.get("reward")
            
            if is_finished:
                sample.status = Sample.Status.COMPLETED
                sample.reward = reward
                break

            # obs_log_probs = [0.0] * len(obs_prompt_ids)
            # _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            # budget = _update_budget(budget, len(obs_prompt_ids))

            # current_image_data = _update_multimodal_state(
            #     sample,
            #     current_image_data,
            #     obs_image_data,
            #     obs_multimodal_inputs,
            #     obs_multimodal_train_inputs,
            #     multimodal_train_inputs_buffer,
            # )

            # if budget is not None and budget <= 0:
            #     sample.status = Sample.Status.TRUNCATED
            #     break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                sample.reward = 0
                break

        # return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass
