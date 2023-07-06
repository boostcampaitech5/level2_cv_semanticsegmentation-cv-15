# import module
import albumentations as A
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from albumentations.pytorch import ToTensorV2
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from segmentation_models_pytorch import UnetPlusPlus
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import XRayDataset
from src.loss import DiceBCELoss
from src.scheduler import CosineAnnealingWarmUpRestarts
from src.utils import set_seed


def data_loader():
    tf = A.Compose(
        [
            A.CLAHE(
                p=1.0,
                clip_limit=(1, 4),
                tile_grid_size=(8, 8),
            ),
            A.Resize(512, 512),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    train_dataset = XRayDataset(
        data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
        transforms=tf,
        split="train",
    )
    val_dataset = XRayDataset(
        data_path="/opt/ml/level2_cv_semanticsegmentation-cv-15/data",
        transforms=tf,
        split="val",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader, val_loader


# objective
def objective(trial):
    # Generate the model.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetPlusPlus(
        encoder_name="tu-hrnet_w64",
        encoder_depth=5,
        encoder_weights="imagenet",
        in_channels=3,
        classes=29,
    ).to(DEVICE)

    # Generate the optimizers.
    # 하이퍼 파라미터들 값 지정
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    T_0 = trial.suggest_int("T_0", 1000, 3000)
    eta_max = trial.suggest_float("eta_max", 0.001, 0.005)
    gamma = trial.suggest_float("gamma", 0.1, 0.5)
    seed = trial.suggest_int("seed", 11, 20)

    set_seed(seed)

    optimizer = getattr(optim, "AdamW")(model.parameters(), lr=0.0001)

    # scheduler
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=T_0,
        T_mult=1,
        eta_max=eta_max,
        T_up=600,
        gamma=gamma,
    )

    criterion = DiceBCELoss()

    for epoch in range(30):
        model.train()
        epoch_loss = 0.0

        for step, (images, masks) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()

            # inference
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            scheduler.step()

        # Validation of the model.
        # Validation에서 얻어진 Score를 활용하여 설정된 Hyperparameter 평가!
        model.eval()
        if (epoch + 1) % 30 == 0:
            dices = []
            with torch.no_grad():
                total_loss = 0
                cnt = 0

                for step, (images, masks) in tqdm(
                    enumerate(valid_loader), total=len(valid_loader)
                ):
                    images, masks = images.cuda(), masks.cuda()
                    model = model.cuda()

                    outputs = model(images)

                    output_h, output_w = outputs.size(-2), outputs.size(-1)
                    mask_h, mask_w = masks.size(-2), masks.size(-1)

                    # restore original size
                    if output_h != mask_h or output_w != mask_w:
                        outputs = F.interpolate(
                            outputs, size=(mask_h, mask_w), mode="bilinear"
                        )

                    loss = criterion(outputs, masks)
                    total_loss += loss
                    cnt += 1

                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > 0.5).detach().cpu()
                    masks = masks.detach().cpu()

                    y_true_f = masks.flatten(2)
                    y_pred_f = outputs.flatten(2)
                    intersection = torch.sum(y_true_f * y_pred_f, -1)

                    eps = 0.0001
                    dices.append(
                        (2.0 * intersection + eps)
                        / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)
                    )

            dices = torch.cat(dices, 0)
            dices_per_class = torch.mean(dices, 0)
            dice_str = [
                f"{c:<12}: {d.item():.4f}"
                for c, d in zip(
                    XRayDataset(
                        "/opt/ml/level2_cv_semanticsegmentation-cv-15/data"
                    ).classes,
                    dices_per_class,
                )
            ]
            dice_str = "\n".join(dice_str)
            print(dice_str)

            avg_dice = torch.mean(dices_per_class).item()

            # Handle pruning based on the intermediate value.
            trial.report(avg_dice, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return avg_dice


if __name__ == "__main__":
    # load data
    train_loader, valid_loader = data_loader()

    # avg_dice 최대가 되는 방향으로 학습을 진행
    study = optuna.create_study(direction="maximize", sampler=TPESampler())

    # n_trials 지정없으면 무한 반복
    study.optimize(objective, n_trials=10)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    trial = study.best_trial

    print("   Value:", {trial.value})

    print("Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
