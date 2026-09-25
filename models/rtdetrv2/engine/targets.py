def prepare_rtdetr_targets(targets):
  """Prepare targets for RT-DETR training."""

  return [
    {
      "class_labels": target["labels"],
      "boxes": target["boxes"]
    }
    for target in targets
  ]