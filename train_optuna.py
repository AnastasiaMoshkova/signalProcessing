import os
import hydra
import logging
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
#from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings('ignore')
from dataset.data_module import DataModule
from solver.solver_reg import Solver
import optuna
from functools import partial
#load_dotenv()
logger = logging.getLogger(__name__)

def objective(trial: optuna.trial.Trial, config) -> float:
    neurons = trial.suggest_int("neurons", 10,30)
    bs = trial.suggest_categorical("batch_size", [16, 32])
    config['model']['config']['linear1'] = neurons
    config['dataset']['train_dataloader']['batch_size'] = bs
    res = nnmain(config)
    return res



def nnmain(config: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    model = instantiate(config.model)
    solver = Solver(model, config)
    model_logger = instantiate(config.logger)
    datamodule = DataModule(config)

    output_dir = hydra_cfg['runtime']['output_dir']
    saving_weight_path = os.path.join(output_dir, "weight")

    checkpoint_callback = ModelCheckpoint(
        dirpath=saving_weight_path,
        save_top_k=5,
        monitor="val/loss"
    )
    early_stopping = EarlyStopping("val/loss")
    trainer = instantiate(
        config.trainer,
        logger=model_logger,
        callbacks=[checkpoint_callback, early_stopping]
    )
    trainer.fit(
        model=solver,
        datamodule=datamodule
    )
    best_val_loss = trainer.callback_metrics["val/best_loss"]

    trainer.test(
        model=solver,
        datamodule=datamodule,
        # ckpt_path=ckpt_path
    )

    return best_val_loss

@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.2.0")
def main(config: DictConfig):
    pruner = (optuna.pruners.MedianPruner())
    study = optuna.create_study(direction = 'minimize', pruner = pruner)

    study.optimize(lambda trial: objective(trial, config), n_trials = 2)

    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trials:')
    trial = study.best_trial

    print("Value: {}".format(trial.value))
    params = {}
    print(" Params: ")
    for key,value in trial.params.items():
        print(" {}: {}".format(key,value))
        params.update({key:value})

if __name__ == "__main__":
    main()
