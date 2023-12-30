import runpy
import os
import sys
from rich import print
def test_use_sde_perforamnces(script:str)->None:
    num_seeds = 5
    for use_sde in [True, False]:
        params: list[str] = [""]
        print(f"Testing use_sde = {use_sde}")
        if use_sde:
            params = params + [f"--use-sde"]
        params = params + [f"--headless"]
        print(f"params: {params}")
        for seed in range(num_seeds):
            sys.argv =   params + [f"--seed={seed}"]
            runpy.run_path(path_name=script, run_name="__main__")

def check_if_reward_the_agent_with_a_normalize_money_amount(script:str)->None:
    num_seeds = 5
    for reward_the_agent_with_a_normalize_money_amount in [True, False]:
        params: list[str] = [""]
        print(f"Testing reward_the_agent_with_a_normalize_money_amount = {reward_the_agent_with_a_normalize_money_amount}")
        if reward_the_agent_with_a_normalize_money_amount:
            params = params + [f"--reward-for-money-amount"]
        params = params + [f"--headless"]
        print(f"params: {params}")
        for seed in range(2, num_seeds):
            sys.argv =   params + [f"--seed={seed}"]
            runpy.run_path(path_name=script, run_name="__main__")

if __name__ == "__main__":
    print(os.getcwd())
    check_if_reward_the_agent_with_a_normalize_money_amount("baselines/baseline_fast_v2.py")
