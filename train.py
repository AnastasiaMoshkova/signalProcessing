import os
import hydra
import logging
from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings
warnings.filterwarnings('ignore')
from dataset.data_module import DataModule
from solver.solver_reg import Solver

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
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

    print(trainer.callback_metrics["val/best_loss"])

    result = trainer.test(
        model=solver,
        datamodule=datamodule,
        # ckpt_path=ckpt_path
    )

    print(result)


if __name__ == "__main__":
    main()
