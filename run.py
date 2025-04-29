import os
import torch
import loguru
import tyro
from datetime import datetime
from omegaconf import OmegaConf
import genesis as gs
from genesis_envs.utils.build_modules import (
    build_module,
    build_optimizer,
    build_agent,
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
    network = build_module(conf["network"])
    if resume_from is not None:
        network.load_state_dict(torch.load(resume_from))
    optimizer = build_optimizer(conf["optimizer"], network)

    agent = build_agent(network=network, optimizer=optimizer, conf=conf["agent"])
    env = build_module(conf["environment"])

    trainer = build_trainer(env, agent, conf)
    if eval:
        while True:
            trainer.rollout()
    else:
        trainer.train_loop()


if __name__ == "__main__":
    tyro.cli(launch)
