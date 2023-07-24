import torch

def window_session(session: torch.Tensor, winsize: int) -> torch.Tensor:
    # Input shape: L x 3
    # Output shape: L x 3*WINSIZE

    # Window session
    x_acc = session[:,0].unsqueeze(1)
    y_acc = session[:,1].unsqueeze(1)
    z_acc = session[:,2].unsqueeze(1)

    w = winsize-1

    xs = [x_acc[:-w]]
    ys = [y_acc[:-w]]
    zs = [z_acc[:-w]]

    for i in range(1,w):
        xs.append(x_acc[i:i-w])
        ys.append(y_acc[i:i-w])
        zs.append(z_acc[i:i-w])

    xs.append(x_acc[w:])
    ys.append(y_acc[w:])
    zs.append(z_acc[w:])

    xs = torch.cat(xs,axis=1).float()
    ys = torch.cat(ys,axis=1).float()
    zs = torch.cat(zs,axis=1).float()

    X = torch.cat([xs,ys,zs], axis=1)
    return X