from models.ode import UniModel


model_fn = {
    'ODE': UniModel,
}


def build_model(model_config):
    model = model_fn[model_config.pop('model_name')](**model_config)
    return model