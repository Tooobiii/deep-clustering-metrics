import inspect
import torch
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from config import MODELS

def build_autoencoder(input_dim, embedding_dim, hidden_layers, activations=torch.nn.LeakyReLU):
    ae_layers = [input_dim] + hidden_layers + [embedding_dim]
    return (FeedforwardAutoencoder, {
        "layers": ae_layers,
        "activation_fn": activations,
    })

def initialize_model(args, data, n_clusters):
    ModelClass = MODELS[args.model]
    sig = inspect.signature(ModelClass)
    DEFAULTS = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    embedding_size = args.embedding_size if args.embedding_size is not None else DEFAULTS["embedding_size"]
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else DEFAULTS["pretrain_epochs"]
    clustering_epochs = args.clustering_epochs if args.clustering_epochs is not None else DEFAULTS["clustering_epochs"]
    architecture = build_autoencoder(input_dim=data.shape[1], embedding_dim=embedding_size,
                                     hidden_layers=args.hidden_layers)
    clustering_loss = args.clustering_loss if args.clustering_loss is not None else DEFAULTS["clustering_loss_weight"]
    pretrain_optimizer = {"lr": args.pretrain_learning_rate} if args.pretrain_learning_rate is not None else DEFAULTS[
        "pretrain_optimizer_params"]
    clustering_optimizer = {"lr": args.clustering_learning_rate} if args.clustering_learning_rate is not None else \
    DEFAULTS["clustering_optimizer_params"]

    model = ModelClass(n_clusters=n_clusters,
                       neural_network=architecture,
                       random_state=args.seed,
                       pretrain_epochs=pretrain_epochs,
                       clustering_epochs=clustering_epochs,
                       clustering_loss_weight=clustering_loss,
                       pretrain_optimizer_params=pretrain_optimizer,
                       clustering_optimizer_params=clustering_optimizer,
                       device="cuda"
                       )
    return model


def extract_architecture(model):
    layers = []
    for name, module in model.neural_network.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers.append(f"{module.in_features}->{module.out_features}")
        elif type(module).__name__ != "Sequential":
            layers.append(type(module).__name__)

    arch_str = " | ".join(layers)

    return arch_str