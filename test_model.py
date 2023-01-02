from torch import load

from scripts.test import test
from main import model_filename, test_dataloader, l

model = load(model_filename)
test(test_dataloader, model, l)