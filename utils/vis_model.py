import torch


def vis_model(model, data, filename="DRN"):
    import onnx.version_converter

    from datetime import datetime as dt
    a = dt.now()
    model_filename = filename + a.strftime("%m%d_%H%M%S") + ".onnx"
    model_filename = "model/checkpoint/" + model_filename

    torch.onnx.export(
        model,
        data,
        model_filename,
        export_params=True,
        opset_version=8
    )

    onnx_model = onnx.load(model_filename)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_filename)
