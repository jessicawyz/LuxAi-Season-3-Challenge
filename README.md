# CSxx46 Project: Lux AI S3

---

## Install Lux S3

    git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3.git
    pip install -e Lux-Design-S3/src
    pip install nvidia-ml-py psutil 
    pip install -U jax[cuda12] jaxlib

Test:

    luxai-s3 --help
    python Lux-Design-S3/src/tests/benchmark_env.py -n 16384 -t 5

If JAX error:

file: ```Lux-Design-S3/src/luxai-s3/env.py```, line ```699```

change:

> scores = (unit_counts_map > 0) & (state.relic_nodes_map_weights <= state.relic_nodes_mask.sum() // 2) & (state.relic_nodes_map_weights > 0)

to:

> scores = (unit_counts_map > 0) & (state.relic_nodes_map_weights.astype('int32') <= state.relic_nodes_mask.sum() // 2) & (state.relic_nodes_map_weights > 0)
               
Run:

    luxai-s3 main.py main.py --output=replay.html

## Dependencies

    cd Lux-Design-S3/kits/python
    git clone https://github.com/jessicawyz/LuxAi-Season-3-Challenge.git
    
    pip install -r requirements.txt

## Usage

    python maddpg.py --use-selfplay --selfplay-ratio 0.5 --num-eval-opponents 5 --num-envs 2 --total-timesteps 10000000 --eval-freq 50000 --snapshot-freq 10000

    luxai-s3 main.py main.py --output=replay.html