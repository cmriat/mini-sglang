from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str], train_config=None) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    from dataclasses import replace as dc_replace

    # If training is configured, defer KV cache init until after training model setup
    if args.train_module:
        args = dc_replace(args, defer_runtime_init=True)

    with torch.no_grad():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        # Set up training module if configured (before KV cache to allow to_empty)
        if args.train_module:
            import gc
            import importlib
            mod = importlib.import_module(args.train_module)
            train_fn = mod.create_train_fn(scheduler, train_config)
            scheduler.set_train_fn(train_fn)
            # Free to_empty temporary allocations, then init KV cache with remaining memory
            gc.collect()
            torch.cuda.empty_cache()
            scheduler.engine.init_runtime()
            scheduler._init_managers()
            init_logger(__name__).info(f"Training module loaded: {args.train_module}")

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    # Shared training config (API server and scheduler both access this)
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    train_config = manager.dict({"lr": 1e-5, "max_grad_norm": 1.0})

    def start_subprocess() -> None:
        from minisgl.tokenizer import tokenize_worker

        world_size = server_args.tp_info.size
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue, train_config),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess, run_shell=run_shell, train_config=train_config)


if __name__ == "__main__":
    launch_server()
