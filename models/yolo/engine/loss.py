from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss


class YOLOLossModelAdapter:
  """Expose the attributes required by Ultralytics v8DetectionLoss."""

  def __init__(self, detect):
    self.detect = detect
    self.model = [detect]
    self.args = get_cfg()

  def parameters(self):
    return self.detect.parameters()


def create_yolo_criterion(model):
  """Create the native YOLOv8 detection loss."""

  adapter = YOLOLossModelAdapter(model.detect)

  return v8DetectionLoss(adapter)