# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import time  

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# the structure of PINN: [2, 256, 256, 256, 256, 256, 256, 256, 1]
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)  
        self.fc5 = nn.Linear(256, 256)  
        self.fc6 = nn.Linear(256, 256) 
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 1)  
        self.sine = Sine()      
        self.relu = nn.ReLU()     

    def forward(self, x):
        x = self.sine(self.fc1(x))
        x = self.sine(self.fc2(x)) 
        x = self.sine(self.fc3(x)) 
        x = self.relu(self.fc4(x)) 
        x = self.relu(self.fc5(x)) 
        x = self.relu(self.fc6(x)) 
        x = self.relu(self.fc7(x))
        x = self.fc8(x) 
        return x


def pde_residual(net, x, k):

    x.requires_grad_(True)
    u = net(x)  # u: (N,1)

    grad_u = torch.autograd.grad(u, x,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]  # (N,2)
    u_x = grad_u[:, 0:1]  # (N,1)
    u_y = grad_u[:, 1:2]  # (N,1)

    u_xx = torch.autograd.grad(u_x, x,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x,
                               grad_outputs=torch.ones_like(u_y),
                               create_graph=True)[0][:, 1:2]
    
    residual = u_xx + u_yy + (k**2) * u
    return residual

def boundary_loss(net, xb, u_exact):

    u_pred = net(xb)
    loss = torch.mean((u_pred - u_exact)**2)
    return loss

k = 50.0

const = np.sin((np.sqrt(2)/2)*k)

N_grid = 302
x_interior = np.linspace(0, 1, N_grid)[1:-1]
y_interior = np.linspace(0, 1, N_grid)[1:-1]
X_mesh, Y_mesh = np.meshgrid(x_interior, y_interior)
X_int = np.hstack((X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)))
X_int = torch.tensor(X_int, dtype=torch.float32)

N_boundary = 302
x_boundary = np.linspace(0, 1, N_boundary)
y_boundary = np.linspace(0, 1, N_boundary)


xb_left = np.hstack((np.zeros((N_boundary, 1)), y_boundary.reshape(-1, 1)))
xb_right = np.hstack((np.ones((N_boundary, 1)), y_boundary.reshape(-1, 1)))
xb_bottom = np.hstack((x_boundary.reshape(-1, 1), np.zeros((N_boundary, 1))))
xb_top = np.hstack((x_boundary.reshape(-1, 1), np.ones((N_boundary, 1))))

xb_left = torch.tensor(xb_left, dtype=torch.float32)
xb_right = torch.tensor(xb_right, dtype=torch.float32)
xb_bottom = torch.tensor(xb_bottom, dtype=torch.float32)
xb_top = torch.tensor(xb_top, dtype=torch.float32)


ub_left = torch.zeros((N_boundary, 1), dtype=torch.float32)
ub_right = const * torch.sin((np.sqrt(2)/2)*k * xb_right[:, 1:2])
ub_bottom = torch.zeros((N_boundary, 1), dtype=torch.float32)
ub_top = const * torch.sin((np.sqrt(2)/2)*k * xb_top[:, 0:1])


t_max = 1e-2
t_min = 1e-4
K_steps = 15000
gamma = (1.0 / K_steps) * np.log(t_max / t_min)

net = PINN()
optimizer = optim.Adam(net.parameters(), lr=t_max)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: np.exp(-gamma * step))


total_losses   = []
interior_losses = []
boundary_losses = []
lr_list        = []


start_time = time.time()

for step in range(K_steps):
    optimizer.zero_grad()

    res = pde_residual(net, X_int, k)
    loss_interior = torch.mean(res**2)

    loss_left   = boundary_loss(net, xb_left, ub_left)
    loss_right  = boundary_loss(net, xb_right, ub_right)
    loss_bottom = boundary_loss(net, xb_bottom, ub_bottom)
    loss_top    = boundary_loss(net, xb_top, ub_top)
    
    loss_boundary = loss_left + loss_right + loss_bottom + loss_top

    loss = loss_interior + loss_boundary
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    total_losses.append(loss.item())
    interior_losses.append(loss_interior.item())
    boundary_losses.append(loss_boundary.item())
    lr_list.append(scheduler.get_last_lr()[0])
    
    if step % 100 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step:5d}: Total Loss = {loss.item():.6e}, LR = {current_lr:.2e}")


end_time = time.time()
training_time = end_time - start_time


net.eval()
with torch.no_grad():
    u_pred_train = net(X_int).cpu().numpy()

X_int_np = X_int.detach().cpu().numpy()
u_exact_train = np.sin((np.sqrt(2)/2)*k * X_int_np[:, 0:1]) * np.sin((np.sqrt(2)/2)*k * X_int_np[:, 1:2])
error_rel_train = np.linalg.norm(u_pred_train - u_exact_train, 2) / np.linalg.norm(u_exact_train, 2)
print(f"Relative L2 error on training data: {error_rel_train:.3e}")


N_test = 152
x_test = np.linspace(0, 1, N_test)
y_test = np.linspace(0, 1, N_test)
X_test, Y_test = np.meshgrid(x_test, y_test)
X_test_flat = np.hstack((X_test.reshape(-1, 1), Y_test.reshape(-1, 1)))
X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)

with torch.no_grad():
    u_pred_test = net(X_test_tensor).cpu().numpy()

u_exact_test = np.sin((np.sqrt(2)/2)*k * X_test_flat[:, 0:1]) * np.sin((np.sqrt(2)/2)*k * X_test_flat[:, 1:2])
error_rel_test = np.linalg.norm(u_pred_test - u_exact_test, 2) / np.linalg.norm(u_exact_test, 2)
print(f"Relative L2 error on test data: {error_rel_test:.3e}")


save_path = 'PINN_k={}'.format(k)
filename = "MGDL_xavier_train{:.4e}.pickle".format(error_rel_train)
fullfilename = os.path.join(save_path, filename)


results = {
    'training_time': training_time,
    'total_losses': total_losses,
    'interior_losses': interior_losses,
    'boundary_losses': boundary_losses,
    'error_rel_train': error_rel_train,
    'error_rel_test': error_rel_test
}
# Ensure that the directory of the save path exists
os.makedirs(save_path, exist_ok=True)


with open(fullfilename, 'wb') as f:
    pickle.dump(results, f)

print(f"Results have been saved to {fullfilename}")
