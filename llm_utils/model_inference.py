import torch
from fastchat.model import load_model, get_conversation_template, add_model_args

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

sampling_params = (
    SamplingParams(temperature=0.6, top_p=0.9, max_tokens=512)
    if SamplingParams is not None
    else None
)


@torch.inference_mode()
def LLM_response(args, model, tokenizer, model_path, prompt, target=False):
    msg = prompt

    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if '70b' in model_path or ('falcon' in model_path) or ('vicuna' in model_path)\
        or ('7b' in model_path and target) \
        or ('mistralai' in model_path) \
        or ('Llama-3' in model_path):
        if sampling_params is None:
            raise ImportError("vllm is required for local vLLM-backed model inference")
        output_ids = model.generate(
            prompts = [prompt],
            sampling_params = sampling_params,
            use_tqdm = False)[0]
        outputs = output_ids.outputs[0].text
        # print(outputs)
    else:
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )

    return outputs

@torch.inference_mode()
def LLM_response_multi(args, model, tokenizer, model_path, prompt_batch):
    prompts = []
    for prompt in prompt_batch:
        msg = prompt

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt_input = conv.get_prompt()
        prompts.append(prompt_input)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompts, padding=True).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    output_ids = output_ids[:, len(input_ids[0]) :]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    return outputs


@torch.inference_mode()
def LLM_response_multi_batch(args, model, tokenizer, model_path, prompt_batch):
    batch_size = 5
    prompts = []
    for prompt in prompt_batch:
        msg = prompt

        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt_input = conv.get_prompt()
        prompts.append(prompt_input)
    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompts, padding=True).input_ids
    # load the input_ids batch by batch to avoid OOM
    outputs = []
    for i in range(0, len(input_ids), batch_size):
        output_ids = model.generate(
            torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
            do_sample=False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        output_ids = output_ids[:, len(input_ids[0]) :]
        outputs.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
    return outputs
