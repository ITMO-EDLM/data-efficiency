from data_efficiency.config import global_config
from data_efficiency.model import ModernBert
from data_efficiency.trainer import Trainer
from data_efficiency.utils import accuracy, f1, upload_dataset


def main() -> None:
    model = ModernBert(global_config.model_name, 2, 0.2, True, False, False)
    validation_dataset = upload_dataset("validation").select(range(100))
    train_dataset = upload_dataset("train").select(range(200))

    trainer = Trainer(
        model,
        "ce",
        {"accuracy": accuracy, "f1": f1},
        val_dataset=validation_dataset,
        train_dataset=train_dataset,
        run_budget=1.0,
        rounds_portions=[0.5, 0.5],
        strategy_data={"strategy_name": "random", "strategy_params": {}},
        optimizer_params={
            "lr": 2e-5,
        },
        n_epochs=3,
        device="mps",
        run_name="random_train_cutted_test_3epoch",
    )
    trainer.setup()
    trainer.run()


if __name__ == "__main__":
    main()
