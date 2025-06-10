import os
from datetime import datetime

import genesis as gs
import loguru
import torch
import tyro
from omegaconf import OmegaConf

from genesis_envs.utils.build_modules import (
    build_agent,
    build_module,
    build_optimizer,
    build_trainer,
)


def launch(
    config_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    exp_name: str | os.PathLike | None = None,
    resume_from: str | os.PathLike | None = None,
    eval: bool = False,
):
    # network → agent ← optimizer
    #             ↓
    #             trainer ← env
    conf = OmegaConf.load(config_path)
    conf.merge_with_cli()

    if eval:
        conf["environment"]["num_envs"] = 1
        conf["environment"]["vis"] = True
        conf["training"]["deterministic_action"] = True
    else:
        conf["training"]["deterministic_action"] = False

    if exp_name is not None:
        conf["training"]["exp_name"] = exp_name
    if output_dir is not None:
        conf["training"]["output_dir"] = output_dir

    if conf["training"].get("output_dir") is not None:
        output_dir = conf["training"].get("output_dir")
        exp_name = conf["training"].get("exp_name")
        logs_dir = os.path.join(output_dir, exp_name, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"{timestamp}.txt")

        loguru.logger.add(
            log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )

    loguru.logger.info("Starting training...")
    loguru.logger.info(f"Config: \n{OmegaConf.to_yaml(conf)}")

    # build genesis init params
    genesis_config = conf.get("genesis", {})
    genesis_init_params = {}
    genesis_init_params["backend"] = getattr(
        gs.constants.backend, genesis_config.get("backend", "cpu")
    )
    genesis_init_params["precision"] = genesis_config.get("precision", "32")
    genesis_init_params["logging_level"] = genesis_config.get("logging_level", None)
    gs.init(**genesis_init_params)

    # build agent params
    networks = {}
    for network_name, network_conf in conf["network"].items():
        networks[network_name] = build_module(network_conf)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=torch.device("cpu"))
        for network_name, network_weights in checkpoint.items():
            networks[network_name].load_state_dict(network_weights)
    optimizer = build_optimizer(conf["optimizer"], networks)

    agent = build_agent(networks=networks, optimizer=optimizer, conf=conf["agent"])
    env = build_module(conf["environment"])

    trainer = build_trainer(env, agent, conf)
    if eval:
        while True:
            trainer.rollout()
    else:
        trainer.train_loop()


if __name__ == "__main__":
    tyro.cli(launch)
