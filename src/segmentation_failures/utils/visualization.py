import seaborn as sns
import torch
import torchvision


def make_image_mask_grid(
    image_batch: torch.Tensor,
    mask_list: torch.Tensor | list[torch.Tensor],
    max_images=-1,
    alpha=0.5,
    slice_idx: list[int] | None = None,
    slice_dim: int = 2,
):
    """Produce image grid from images and predictions.

    Args:
        image_batch (torch.Tensor): shape [batch, modality, *spatial_dims]
        pred_batch_list (torch.Tensor): list of tensors with shape [batch, class, *spatial_dims], one hot encoded!
        max_images (int, optional): limit number of images in output. Defaults to 5.
        alpha (float, optional): alpha value for mask overlay. Defaults to 0.5.
        slice_idx (list[int], optional): slice index for 3D images. Defaults to None.
        slice_dim (int, optional): dimension to slice for 3D images (only counting spatial dims). Defaults to 2.

    Returns:
        torch.Tensor: RGB image grid
    """
    if max_images == -1:
        max_images = image_batch.shape[0]
    if not isinstance(mask_list, (list, tuple)):
        mask_list = [mask_list]
    all_data = [image_batch] + mask_list
    for i, batch_data in enumerate(all_data):
        batch_data = batch_data.detach().cpu()
        batch_data = batch_data[:max_images]
        if batch_data.ndim == 5:
            # for 3D just take one slice
            if slice_idx is None:
                slice_idx = [batch_data.shape[-1] // 2] * batch_data.shape[0]
            # if the slice_idx is a list, select slice slice_idx[i] for batch i
            assert isinstance(slice_idx, (list, tuple))
            batch_data = torch.stack(
                [
                    batch.select(dim=1 + slice_dim, index=slice_idx[i])
                    for i, batch in enumerate(batch_data)
                ]
            )
        if i > 0:
            batch_data = batch_data.to(dtype=torch.bool)
        all_data[i] = batch_data
        # NOTE I could remove the background class by doing pred_batch = pred_batch[1:]
        # but then I would need to define colors (the torchvision function color palette starts with black)
    image_batch, mask_list = all_data[0], all_data[1:]
    # pick only first modality in image# roi is binary mask
    image_batch = image_batch[:, 0]
    # normalize image and convert to RGB
    tmp_min = image_batch.flatten(start_dim=1).min()
    tmp_max = image_batch.flatten(start_dim=1).max()
    image_batch = (image_batch - tmp_min) / (tmp_max - tmp_min)
    image_batch = torch.stack([image_batch, image_batch, image_batch], dim=1)
    if image_batch.is_floating_point():
        image_batch = (image_batch * 255).to(dtype=torch.uint8)
    grid_list = []

    for idx, img in enumerate(image_batch):
        grid_list.append(img)
        for mask in mask_list:
            colors = sns.color_palette(n_colors=mask.shape[1]).as_hex()
            colors.insert(0, "#000000")
            grid_list.append(
                # torchvision.utils.draw_segmentation_masks(img, mask[idx], alpha=alpha)
                torchvision.utils.draw_segmentation_masks(
                    img, mask[idx], alpha=alpha, colors=colors
                )
            )
    return torchvision.utils.make_grid(grid_list, nrow=len(mask_list) + 1, normalize=False)
