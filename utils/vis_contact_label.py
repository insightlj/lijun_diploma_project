import pandas as pd
import seaborn as sns
import torch

from merge_dim import merge_dim

def vis_contact(label, is_pred=False):
    import matplotlib.pyplot as plt

    if is_pred:
        merge_dim(label) # [5,5,2] -> [5,5]

    df = pd.DataFrame(label.numpy())
    sns.heatmap(df, cmap="YlGnBu", cbar=False)
    plt.show()





if __name__== "__main__":
    # contact_label: [5,5]
    contact_label = torch.tensor(([0, 1, 0, 1, 1],
                                  [0, 0, 0, 1, 1],
                                  [0, 0, 0, 1, 0],
                                  [1, 1, 1, 0, 1],
                                  [0, 1, 0, 1, 0]))

    # pred: [5,5,2]
    pred = contact_label.unsqueeze(dim=2)
    pred = torch.concat((pred, pred), dim=2)
    pred = (pred + 0.8) / 2

    vis_contact(pred, is_pred=True)
    vis_contact(contact_label, is_pred=False)

    fake_contact = torch.zeros((192, 192))

    import random
    for i in range(200):
        a = random.randint(0,191)
        b = random.randint(0,191)
        fake_contact[a,b] = 1

    vis_contact(fake_contact)

