#! /cpfs01/user/liupengfei/jlli/anaconda3/envs/llama_factory/bin/python

import sys
interpreter_path = sys.executable
print("Python Interpreter Path:", interpreter_path)

from llmtuner import run_exp



def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
