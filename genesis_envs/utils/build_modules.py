import torch
from ..runners import GenesisEnvTrainer
from .config_utils import import_str


def build_module(conf):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")
    module = module_cls(**module_params)
    return module

def build_optimizer(conf, network):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")
    module = module_cls(network.parameters(), **module_params)
    return module

def build_agent(conf, network, optimizer):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")
    module = module_cls(network=network, optimizer=optimizer, **module_params)
    return module

def build_trainer(env, agent, conf):
    trainer = GenesisEnvTrainer(
        env=env,
        agent=agent,
        **conf["training"],
    )
    return trainer
