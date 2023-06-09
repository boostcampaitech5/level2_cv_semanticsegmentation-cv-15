import hydra

from src.config import register_configs
from src.train import train


@hydra.main(config_name="default", version_base="1.3")
def main(config):
    train(config=config)


if __name__ == "__main__":
    register_configs()
    main()
