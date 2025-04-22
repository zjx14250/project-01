from .mixer import StockMixer
from .transformer import StockTransformer
from .Attraos import Attraos

def get_model(model_name, **kwargs):
    if model_name == "StockMixer":
        print("Model: StockMixer")
        return StockMixer(**kwargs)
    elif model_name == "StockTransformer":
        # 只取StockTransformer需要的参数
        print("Model: StockTransformer")
        keys = ['stocks', 'time_steps', 'channels', 'nhead', 'num_layers', 'dim_feedforward', 'dropout']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in keys}
        return StockTransformer(**filtered_kwargs)
    elif model_name == "Attraos":
        print("Model: Attraos")
        # 直接使用传入的config对象
        config = kwargs.get('config')
        if config is None:
            raise ValueError("Attraos model requires a config object")
        return Attraos(config)
    else:
        raise ValueError(f"Unknown model: {model_name}") 