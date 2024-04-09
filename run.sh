torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-S/2 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/s_2

torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-S/4 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/s_4

torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-B/2 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/b_2

torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-B/4 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/b_4

torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-L/2 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/l_2

torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-L/4 --data-path /home/ubuntu/data/datasets/imangenet1k --features-path /home/ubuntu/data/datasets/dit_features/l_4