from ..runners import GenesisEnvTrainer
from .config_utils import import_str


def build_module(conf):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")
    module = module_cls(**module_params)
    return module


def build_optimizer(conf, networks):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")

    params = []
    for network in networks.values():
        params.append({"params": network.parameters(), **module_params})
    module = module_cls(params, **module_params)
    return module


def build_agent(conf, networks, optimizer):
    module_cls = import_str(conf["type"])
    module_params = conf.copy()
    module_params.pop("type")
    module = module_cls(networks=networks, optimizer=optimizer, **module_params)
    return module


def build_trainer(env, agent, conf):
    trainer = GenesisEnvTrainer(
        env=env,
        agent=agent,
        **conf["training"],
    )
    return trainer
