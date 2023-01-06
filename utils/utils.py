import torch

# save model parameters
def save_model_parameters(model, time, filename = "resnet_li"):

    model_filename = filename + time.strftime("%m%d_%H%M%S") + ".pt"
    model_filename = "model/checkpoint/" + model_filename

    torch.save(model, model_filename)

    return model_filename