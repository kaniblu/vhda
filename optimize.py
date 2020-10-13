__all__ = ["optimize"]

import datetime
import tempfile
import pathlib
import warnings
import itertools
import functools
from dataclasses import dataclass
from typing import Callable

import yaap
import optuna

import utils


def find_replace_kvp(d: dict, key: str, value):
    if key in d:
        d[key] = value
        return
    for k, v in d.items():
        if isinstance(v, dict):
            find_replace_kvp(v, key, value)


def update_dict(d: dict, key: str, value):
    tokens = key.split(".")
    if tokens[0] not in d:
        raise KeyError(f"not a valid key: {tokens[0]}")
    if len(tokens) == 1:
        d[tokens[0]] = value
        return
    update_dict(d[tokens[0]], ".".join(tokens[1:]), value)


def update_model(model: dict, key: str, value):
    update_dict(
        d=model,
        key=".".join(itertools.chain(*zip(itertools.repeat("vargs"),
                                          key.split(".")))),
        value=value
    )


@dataclass
class Optimizer:
    trial: optuna.Trial

    def suggest_dim(self, name: str, low: int, high: int):
        bounds = (low, high)
        pow_map = {2 ** i: i for i in range(1, 11)}
        if not all(b in pow_map for b in bounds):
            raise ValueError(f"dimension bounds must be power of 2: {bounds}")
        return 2 ** self.trial.suggest_int(
            name=name,
            low=pow_map[bounds[0]], high=pow_map[bounds[1]]
        )

    def suggest_multilayer(self, name: str) -> dict:
        return dict(
            type="multilayer",
            vargs=dict(
                activation=self.trial.suggest_categorical(
                    name=f"multilayer-act",
                    choices=("relu", "tanh")
                ),
                batch_norm=self.trial.suggest_categorical(
                    name=f"multilayer-bn",
                    choices=(False, True)
                ),
                dropout=self.trial.suggest_discrete_uniform(
                    name=f"multilayer-dropout",
                    low=0.0, high=0.40, q=0.1
                ),
                hidden_dim=2 ** self.trial.suggest_int(
                    name=f"multilayer-dim-pow",
                    low=8, high=9
                ),
                num_layers=self.trial.suggest_int(
                    name=f"multilayer-num-layers",
                    low=1, high=2
                )
            )
        )

    def optimize_rnn(self, model, key: str):
        optuna_key = '-'.join(key.split('.'))
        update_model(
            model=model,
            key=f"{key}.dropout",
            value=self.trial.suggest_discrete_uniform(
                name=f"{optuna_key}-dropout",
                low=0.0, high=0.40, q=0.1
            )
        )
        try:
            update_model(model, f"{key}.num_layers", self.trial.suggest_int(
                name=f"{optuna_key}-num-layers",
                low=1, high=2
            ))
        except KeyError:
            update_model(model, f"{key}.layers", self.trial.suggest_int(
                name=f"{optuna_key}-num-layers",
                low=1, high=2
            ))

    def optimize_posterior_dropout(self, model: dict):
        dropout_gamma = self.trial.suggest_discrete_uniform(
            name="posterior-dropout-gamma",
            low=0.9, high=1.3, q=0.05)
        base_dropout = self.trial.suggest_discrete_uniform(
            name="posterior-dropout-base",
            low=0.01, high=0.1, q=0.01
        )
        speaker_dropout = base_dropout
        goal_dropout = speaker_dropout * dropout_gamma
        turn_dropout = goal_dropout * dropout_gamma
        sent_dropout = turn_dropout * dropout_gamma
        word_dropout = sent_dropout * dropout_gamma
        update_model(model, "speaker_dropout", speaker_dropout)
        update_model(model, "goal_dropout", goal_dropout)
        update_model(model, "turn_dropout", turn_dropout)
        update_model(model, "sent_dropout", sent_dropout)
        update_model(model, "sent_decoder.word_dropout", word_dropout)

    def optimize_dim(self, model, key, low, high):
        update_model(
            model=model,
            key=f"{key}-pow",
            value=self.suggest_dim(
                name="-".join(key.split(".")),
                low=low,
                high=high
            )
        )

    def optimize_multilayer(self, model, key):
        update_model(
            model=model,
            key=key,
            value=self.suggest_multilayer("-".join(key.split(".")))
        )

    def optimize_model(self, model: dict):
        self.optimize_posterior_dropout(model)
        self.optimize_dim(model, "zconv_dim", 4, 32)
        self.optimize_dim(model, "zgoal_dim", 8, 64)
        self.optimize_dim(model, "zturn_dim", 8, 64)
        self.optimize_dim(model, "zutt_dim", 64, 256)
        self.optimize_dim(model, "conv_dim", 128, 512)
        self.optimize_dim(model, "goal_dim", 64, 256)
        self.optimize_dim(model, "turn_dim", 64, 256)
        self.optimize_dim(model, "sent_dim", 128, 512)
        self.optimize_dim(model, "sent_encoder.hidden_dim", 256, 1024)
        self.optimize_rnn(model, "sent_encoder.rnn")
        self.optimize_rnn(model, "conv_encoder")
        self.optimize_rnn(model, "conv_post_encoder")
        self.optimize_rnn(model, "sent_decoder.decoding_rnn")
        for key in (
                ("sent_encoder", "output_layer"),
                ("sent_decoder", "output_layer"),
                ("sent_decoder", "decoding_rnn", "init_layer"),
                ("state_encoder", "label_layer"),
                ("state_encoder", "output_layer"),
                ("state_decoder", "input_layer"),
                ("state_decoder", "output_layer"),
                ("speaker_encoder",),
                ("speaker_decoder",)
        ):
            self.optimize_multilayer(model, ".".join(key))
        return model

    def optimize_config(self, config: dict):
        gen_epoch = int(self.trial.suggest_discrete_uniform(
            name="gen-epoch",
            low=500, high=1000, q=100
        ))
        dropout_annealing_target = self.trial.suggest_discrete_uniform(
            name="dropout-annealing-target",
            low=0.0, high=0.2, q=0.02
        )
        dropout_annealing_period = int(self.trial.suggest_discrete_uniform(
            name="dropout-annealing-period",
            low=250000, high=400000, q=50000
        ))
        dropout_annealing = \
            [(0, 1.0), (dropout_annealing_period, dropout_annealing_target)]
        kld_annealing_period = int(self.trial.suggest_discrete_uniform(
            name="kld-annealing-period",
            low=150000, high=250000, q=100000
        ))
        base_gen_scale = self.trial.suggest_discrete_uniform(
            name="base-gen-scale",
            low=0.2, high=2.0, q=0.2
        )
        gen_scale_gamma = self.trial.suggest_discrete_uniform(
            name="gen-scale-gamma",
            low=0.7, high=1.3, q=0.05
        )
        vhda_conv_scale = base_gen_scale
        vhda_speaker_scale = 0.0
        vhda_goal_scale = vhda_conv_scale * gen_scale_gamma
        vhda_turn_scale = vhda_goal_scale * gen_scale_gamma
        vhda_sent_scale = vhda_turn_scale * gen_scale_gamma
        config["batch-size"] = 32
        config["num-vhda-epochs"] = gen_epoch
        config["save-every"] = gen_epoch
        config["dropout-schedule"] = str(dropout_annealing)
        config["kld-annealing-rate"] = kld_annealing_period
        config["vhda-conv-scale"] = vhda_conv_scale
        config["vhda-speaker-scale"] = vhda_speaker_scale
        config["vhda-goal-scale"] = vhda_goal_scale
        config["vhda-turn-scale"] = vhda_turn_scale
        config["vhda-sent-scale"] = vhda_sent_scale
        return config


def optimize(trial: optuna.Trial, model_path, config_path):
    optimizer = Optimizer(trial)
    run_config = utils.load_yaml(config_path)
    mdl_config = utils.load_yaml(model_path)
    run_config = optimizer.optimize_config(run_config)
    mdl_config = optimizer.optimize_model(mdl_config)
    shell = utils.ShellUtils()
    shell.mkdir("optimize-debug", silent=True)
    utils.save_yaml(mdl_config, "optimize-debug/model.yml")
    utils.save_json(run_config, "optimize-debug/run.json")
    run_path, mdl_path = tempfile.mktemp(), tempfile.mktemp()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = (pathlib.Path(__file__).absolute()
                .parent.joinpath(f"out/woz/{timestamp}"))
    run_config["save-dir"] = str(save_dir)
    run_config["model-path"] = mdl_path
    utils.save_json(run_config, run_path)
    utils.save_json(mdl_config, mdl_path)
    retcode, stdout, stderr = utils.Process(
        args=f"python run.py @load {run_path}".split(),
        cwd=pathlib.Path(__file__).absolute().parent,
        print_stdout=True,
        print_stderr=True
    ).run()
    if retcode:
        raise RuntimeError(f"process 'run.py' failed; "
                           f"return code: {retcode}; stderr: {stderr}")
    shell.remove(run_path, silent=True)
    shell.remove(mdl_path, silent=True)
    gen_dirs = list(save_dir.glob("gen-*"))
    if not gen_dirs:
        raise RuntimeError(f"no generation directory detected")
    if len(gen_dirs) > 1:
        warnings.warn(f"more than 1 generation "
                      f"directories detected: {gen_dirs}")
    gen_dir = gen_dirs[-1]
    ttest_results = utils.load_json(gen_dir.joinpath("ttest-results.json"))
    return -ttest_results["hmean"]["t"]


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("examples/model-vhda.yml")),
                   help="Path to a base model path.")
    parser.add_pth("run-path", must_exist=True, required=True,
                   help="Path to a base run configuration path.")
    parser.add_str("storage", format="url",
                   default="sqlite:///examples/study.db",
                   help="Optuna database url supported by sqlalchemy.")
    parser.add_str("study-name", default="default",
                   help="Optuna study name.")
    parser.add_int("num-trials",
                   help="Number of trials.")
    parser.add_int("num-jobs", default=1,
                   help="Number of concurrent jobs.")
    parser.add_flt("timeout",
                   help="Timeout for a single trial in seconds.")
    return parser


def main():
    parser = create_parser()
    args = utils.parse_args(parser)
    study = optuna.create_study(
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True
    )
    optimize_fn: Callable[[optuna.Trial], float] = functools.partial(
        optimize,
        model_path=args.model_path,
        config_path=args.run_path
    )
    study.optimize(
        func=optimize_fn,
        n_trials=args.num_trials,
        n_jobs=args.num_jobs,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()
