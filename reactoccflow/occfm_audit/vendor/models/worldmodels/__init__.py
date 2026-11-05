from reactoccflow.occfm_audit.vendor.models.model_template import ModelTemplate
from reactoccflow.occfm_audit.vendor.models.worldmodels.occfm import OccFM

__all__ = {
    'ModelTemplate' : ModelTemplate,
    'OccFM': OccFM,
}


def build_wm(model_cfg, loss_cfg, cache_mode):

    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, loss_cfg=loss_cfg, cache_mode=cache_mode
    )
    return model
