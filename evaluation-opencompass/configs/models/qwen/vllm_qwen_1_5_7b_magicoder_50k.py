

from opencompass.models import VLLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=151645,
)

models = [
    dict(
        type=VLLM,
        abbr='qwen1.5-7b-magicoder-50k-vllm',
        path="/cpfs01/shared/GAIR/GAIR_hdd/yguo/ckpts/qwen1.5-7b-magicoder-50k/",
        model_kwargs=dict(tensor_parallel_size=4),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        end_str='<|im_end|>',
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]
