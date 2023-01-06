from torch import load

from scripts.test import test
from main import model_filename, test_dataloader

model = load(model_filename)
test(test_dataloader, model)