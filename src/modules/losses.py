import torch.nn.functional as F


def hinge_loss_dis(fake, real, lbl_real):
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape
    loss = F.relu(1.0 - real).mean() + \
           F.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake, lbl_fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss
