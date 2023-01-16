import torch

# save model parameters
def save_model_parameters(model, time, filename = "resnet_li"):

    model_filename = filename + "_" + time.strftime("%m%d_%H%M") + ".pt"
    model_filename = "model/checkpoint/" + "pt_" + time.strftime("%m%d") + "/" + model_filename

    torch.save(model, model_filename)