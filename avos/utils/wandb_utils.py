import pathlib
from typing import Optional, Dict, List
import wandb
import torch
import numpy as np
import random
# import avos.utils.misc as misc


def init_or_resume_wandb_run(wandb_id_file_path: pathlib.Path,
                             project_name: Optional[str] = None,
                             entity_name: Optional[str] = None,
                             run_name: Optional[str] = None,
                             config: Optional[Dict] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.
        Returns the config, if it's not None it will also update it first
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        print('Resuming from wandb path... ', wandb_id_file_path)
        resume_id = wandb_id_file_path.read_text()
        wandb.init(entity=entity_name,
                   project=project_name,
                   name=run_name,
                   resume=resume_id,
                   config=config)
        # settings=wandb.Settings(start_method="thread"))
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(entity=entity_name, project=project_name, name=run_name, config=config)
        wandb_id_file_path.write_text(str(run.id))

    wandb_config = wandb.config
    if config is not None:
        # update the current passed in config with the wandb_config
        wandb.config.update(config)

    return config


def get_viz_img(images: torch.tensor, targets: List[Dict], outputs: Dict[str, torch.tensor],
                itr, soenet_feats=None):
    """
    Generate Image for visualization
    Args:
        images: [T x C x H x W]
        targets: [{'masks': torch.tensor [T x H x W], ...}]
        outputs: {'pred_masks': torch.tensor [B x T x h x w] }
        itr: Inverse transform
        soenet_feats: {layer: torch.tensor}
    """
    src_masks = outputs["pred_masks"]

    # Prepare Groundtrh masks
    target_masks = [t["masks"] for t in targets]
    target_masks, valid = utils.nested_tensor_from_tensor_list(target_masks, split=False).decompose()
    temp_target_masks = target_masks[0].cpu().numpy()
    target_masks = target_masks[0].cpu().unsqueeze(3).repeat(1, 1, 1, 3) * 255

    # Generate Predicted Masks with Thresholding
    src_masks = utils.interpolate(src_masks, size=temp_target_masks.shape[-2:], mode="bilinear", align_corners=False)
    src_masks = src_masks[0].cpu().detach().numpy()
    yc = src_masks.copy()
    batch_gt = temp_target_masks.copy()
    yc_max_hwt = np.transpose(yc, [1, 2, 0])
    gt_mask_hwt = np.transpose(batch_gt, [1, 2, 0])

    bin_mask_hwt = np.ones_like(yc_max_hwt)
    bin_mask_hwt[yc_max_hwt<0.5] = 0


    pred_masks = torch.tensor(bin_mask_hwt).permute(2, 0, 1).unsqueeze(3).repeat(1, 1, 1, 3)
    pred_masks = pred_masks.numpy() * 255

    # Inverse Transform Images to Denormalize
    T, C, H, W = images.shape
    images_itr = itr(images.clone())
    images_itr = images_itr.cpu().numpy() * 255
    images_itr = images_itr.astype(np.uint8)
    images_itr = np.transpose(images_itr, [0, 2, 3, 1])

    images = images.cpu().numpy()
    images = images.astype(np.uint8)
    images = np.transpose(images, [0, 2, 3, 1])
    batch_size = images.shape[0]

    # Create concatenated Image for randomly sampled N frames
    nframes = 3
    rnd_idx = random.randint(0, T - nframes - 1)
    viz_cat_img = []
    for i in range(nframes):
        cat_img = np.concatenate((images_itr[rnd_idx + i], target_masks[rnd_idx + i], pred_masks[rnd_idx + i]),
                                 axis=1)
        viz_cat_img.append(cat_img)
    viz_cat_img = np.concatenate(viz_cat_img, axis=0)

    return viz_cat_img
