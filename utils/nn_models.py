"""
Neural network models to be used in channel charting.
First, define a generic fully connected network. Then, define "model" classes to train and evaluate the network.

For channel charting for position and orientation, the neural network training is extended to include the novel
triplet-loss for orientation as well as the novel alignment loss for absolute orientation training. 
For neural positioning, the neural network training is extended by a cosine dissimilarity loss for joint 
position and orientation training.

@author: Sueda Taner, modified by Daniel Richner
"""
#%%
import numpy as np
import torch
import torch.optim
from torch import nn
from tqdm import trange
from torch.nn.functional import triplet_margin_loss, triplet_margin_with_distance_loss
from torchinfo import summary

import typing
import time
from matplotlib import pyplot as plt
from operator import itemgetter

from utils.loss_helpers import get_pos_neg_edges, get_triplet_batch, bb_loss

#%% The class for a generic fully connected network to be used in all methods

class FCNet(nn.Module):
    """
    A class for a generic fully connected network.
    This network is the building block for all channel charting methods.
    
    """
    
    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int, add_BN=False):
        """
        Create a fully connected network.

        Parameters
        ----------
        in_features : int
            Input features' dimension.
        hidden_features : typing.Tuple[int, ...]
            Tuple where each entry corresponds to a hidden layer with the 
            corresponding feature dimension.
        out_features : int
            Output features' dimension.
        """
        super().__init__()
        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1

        self.layers = nn.ModuleList()
        for idx in range(num_affine_maps):
            self.layers.append(nn.Linear(feature_sizes[idx], feature_sizes[idx + 1]))
            if idx < num_affine_maps-1:
                torch.nn.init.kaiming_normal_(self.layers[-1].weight, mode='fan_in', nonlinearity='relu')
            else: 
                torch.nn.init.xavier_normal_(self.layers[-1].weight) # Glorot initialization
            
                
            if add_BN and idx == 0:# idx != num_affine_maps - 1:
                self.layers.append(nn.BatchNorm1d(feature_sizes[idx + 1]))
        self.num_layers = len(self.layers)

        self.activation = nn.ReLU()
        self.add_BN = add_BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fully connected network.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        current_features : torch.Tensor
            Output of the network.

        """
        current_features = x
        for idx, current_layer in enumerate(self.layers[:-1]):
            current_features = current_layer(current_features)
            if not self.add_BN: current_features = self.activation(current_features)
            else:
                if idx > 0: # activation after BN
                    current_features = self.activation(current_features)
        
        # Last layer has no BN or activation
        current_features = self.layers[-1](current_features) # last layer has no activation
       
        return current_features    
     

#%% Functions that are shared among multiple classes of models

def triplet_pass(model, X: torch.Tensor, triplet_batch: np.array):
    """
    Given a model, a dataset, and the triplet indices from this dataset,
    pass the triplets through the model's network and compute the loss.

    Parameters
    ----------
    model : SelfOrWeaklySupervisedModel or SemisupervisedModel
        The model's network passes the triplets.
        The loss is computed according to the model's parameters.
    X : torch.Tensor of size (num_samples, feature_dimension)
        Dataset.
    triplet_batch : np.array of size (num_anchors,num_triplets_per_anchor,num_triplets_per_anchor)
        Indices of anchor-pos sample-neg sample triplets. 
        
    Returns
    -------
    batch_y : torch.Tensor
        The output of the network.
    mapper : np.array
        A mapper between the dataset indices in X and the batch
    dataset_idcs : np.array
        The indices of all samples passed through the network.
    loss : torch.Tensor
        Triplet loss for the given triplet batch.

    """
    # Remove duplicates for forward pass
    dataset_idcs = np.unique(triplet_batch.flatten())
    
    # Extract the individual samples
    batch_x = X[dataset_idcs].to(model.device)
    
    # Map from the dataset index to the index in the current batch
    mapper = torch.zeros((np.max(dataset_idcs) + 1), dtype=int, device=model.device)
    mapper[dataset_idcs] = torch.arange(batch_x.shape[0], dtype=int, device=model.device) 
    
    # Make a forward pass
    output = model.network(batch_x)
    batch_y_pos, batch_y_rot = output[:, :2], output[:, 2:]
    # Calculate the triplet loss
    y_a_pos = batch_y_pos[mapper[triplet_batch[:, 0]]] # anchor samples
    y_c_pos = batch_y_pos[mapper[triplet_batch[:, 1]]] # positive (close) samples
    y_f_pos = batch_y_pos[mapper[triplet_batch[:, 2]]] # negative (far) samples
    
    y_a_rot = batch_y_rot[mapper[triplet_batch[:, 0]]] # anchor samples
    y_c_rot = batch_y_rot[mapper[triplet_batch[:, 1]]] # positive (close) samples
    y_f_rot = batch_y_rot[mapper[triplet_batch[:, 2]]] # negative (far) samples

    loss_position = triplet_margin_loss(y_a_pos, y_c_pos, y_f_pos, margin=model.par.M_t)

    def angular_distance(v1, v2):
        # v1, v2 live in R^2 and represent (sin phi, cos phi).
        # After L2-normalisation the dot product equals cos(phi_1 - phi_2),
        # so 1 - dot is exactly the 1 - cos(Delta phi) loss but
        # without ever materialising an angle.
        v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
        v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
        return 1.0 - (v1 * v2).sum(dim=-1, keepdim=True)

    M_r = getattr(model.par, 'M_r', 1.0)
    loss_rotation = triplet_margin_with_distance_loss(y_a_rot, y_c_rot, y_f_rot, margin=M_r, distance_function=angular_distance)

    # Motion direction alignment loss: for every anchor-positive (close) pair (which
    # is close in time, so the chord in predicted position space is
    # approximately tangent to the trajectory), we require the predicted
    # heading vector to align with that chord direction.
    # This is NOT sign-symmetric (reversing v gives loss = 2), so it breaks
    # the reflection degeneracy of the cosine-based triplet rotation loss
    # above. The chord is detached so this loss trains only the rotation
    # head, never the position head.
    loss_align = torch.zeros((), device=batch_y_pos.device)
    a_idx_np = triplet_batch[:, 0]
    c_idx_np = triplet_batch[:, 1]
    chord_sign_np = np.where(c_idx_np >= a_idx_np, 1.0, -1.0).astype(np.float32)
    chord_sign = torch.from_numpy(chord_sign_np).to(model.device).unsqueeze(-1)
    chord = (chord_sign * (y_c_pos - y_a_pos).detach())
    chord_norm = chord.norm(dim=-1, keepdim=True)
    min_chord = getattr(model.par, 'align_min_chord', 1e-3)
    valid = (chord_norm.squeeze(-1) > min_chord)
    if valid.any():
        chord_hat = chord[valid] / (chord_norm[valid] + 1e-8)
        v_a = y_a_rot[valid]
        v_c = y_c_rot[valid]
        v_a = v_a / (v_a.norm(dim=-1, keepdim=True) + 1e-8)
        v_c = v_c / (v_c.norm(dim=-1, keepdim=True) + 1e-8)
        v_mid = 0.5 * (v_a + v_c)
        v_mid = v_mid / (v_mid.norm(dim=-1, keepdim=True) + 1e-8)
        loss_align = (1.0 - (chord_hat * v_mid).sum(dim=-1)).mean()

    # Output all passed samples for potential use in bilateration loss
    return batch_y_pos, batch_y_rot, mapper, dataset_idcs, loss_position, loss_rotation, loss_align    

def predict(model, X: torch.Tensor) -> torch.Tensor:
    """
    Set a model's network in evaluation mode and make a forward pass.
    24-01-26 updates: Y is not in GPU device anymore. Added with torch.no_grad()

    Parameters
    ----------
    X : torch.Tensor
        Dataset (i.e, the CSI features).

    Returns
    -------
    X : torch.Tensor
        Channel chart.

    """        
    model.network.eval()
    
    test_loader = torch.utils.data.DataLoader(
        X, batch_size=model.par.batch_size, shuffle=False) 
    
    Y_pos = torch.zeros((0, 2))
    # Channel charting uses (sin phi, cos phi).
    Y_rot = torch.zeros((0, 2))
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(model.device)
            batch_y = model.network(batch_x)
            batch_y_pos, batch_y_rot = batch_y[:, :2], batch_y[:, 2:]
            Y_pos = torch.concatenate((Y_pos, batch_y_pos.detach().cpu()))
            Y_rot = torch.concatenate((Y_rot, batch_y_rot.detach().cpu()))
    return Y_pos, Y_rot

def set_model_device(model, device):
    """
    Set the model's device and move the model's network to this device.

    Parameters
    ----------
    model : SemiOrWeaklySupervisedModel or SupervisedModel or SemisupervisedModel 
        A ...Model object.
    device : 
        torch.device or "cpu"

    Returns
    -------
    None.

    """
    model.device = device
    model.network.to(device)
    
#%% "Models" to train the FCNet using different learning approaches

class TripletModel:
    """
    A class to train an FCNet using triplet loss and / or bilateration loss.
    
    Training with the triplet loss = self-supervised learning, and 
    training with the bilaterion loss = weakly-supervised learning. 
    
    Attributes
    ----------
    device : torch.device
        Device to use for training, e.g. torch.device("cpu") or torch.device("cuda:0").
    par : Parameter
        Simulation scenario and training-related parameters are all stored in this object.
    
    """
    
    def __init__(self, device, par):
    
        self.device = device # training device
        self.par = par # training parameters

        # The model controls the network
        self.network = FCNet(self.par.in_features, self.par.hidden_features, self.par.out_features)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.par.learning_rate)
        self.network.to(device)
        if par.use_scheduler == 1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, par.scheduler_param) 

        
    def print_params(self):
        # Print training parameters
        if self.par.use_scheduler == 0:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t, ', M_p:', self.par.M_p, self.par.P_thresh,
              'use_sch:', self.par.use_scheduler)   
        else:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t, ', M_p:', self.par.M_p, self.par.P_thresh,
              'use_sch:', self.par.use_scheduler, ', sche param:', self.par.scheduler_param)
    
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        bounding_boxes=None,
        box_labels=None,
        plot_each_loss=False,
    ):
        """
        Train the network based on model parameters.
        If lambda_b = 0, the training is based on only the triplet loss -> self-supervised.
        Else if lambda_b = inf, the training is based on only the bilateration loss -> weakly-supervised.
        Else -> self- and weakly-supervised.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Training dataset consisting of CSI features, timestamps, and 
            normalized received power per AP. (The best AP for each UE should have 0 dB.)
            

        Returns
        -------
        loss_per_epoch : np.array
            Average loss per batch at each epoch for training visualization.

        """
        X = dataset[:][0] 
        timestamps = dataset[:][1]
        pow_per_ap = dataset[:][2]

        if self.par.lambda_box: 
            assert bounding_boxes is not None
            bounding_boxes = bounding_boxes.to(self.device)
            if self.par.bb_src != 'ap': # If the source of the boxes are not APs, box labels must be given for each UE
                assert box_labels is not None
                assert len(box_labels) == X.shape[0] 
                
        # Make sure to use only the first D entries of the APs' positions
        ap_pos = self.par.ap_pos[:,:self.par.out_features] # potentially exclude the z axis
        ap_pos = torch.tensor(ap_pos).to(self.device)
        
        # Initialize the set of UE - high powered AP - low powered AP triplets
        u_ap_ap_triplets = [torch.zeros((3,0))] * X.shape[0] # for bilateration loss
        u_ap_pairs = [torch.zeros((2,0))] * X.shape[0] # for box loss
        valid_APs_per_u = [[]]*X.shape[0]
        num_valid_APs_per_u = np.zeros(X.shape[0], dtype=int)
        # We won't include APs whose received power is < P_thresh
        thresh = self.par.P_thresh # to exclude APs whose power is below this
        for u in range(X.shape[0]):
            valid_APs = torch.where(pow_per_ap[u] > thresh)[0]
            valid_APs_per_u[u] = valid_APs
            num_valid_APs_per_u[u] = len(valid_APs)
            if num_valid_APs_per_u[u] > 0:
                # For box loss:
                if self.par.box_mode == 1:
                    u_ap_pairs[u] = torch.tensor([[u], [torch.argmax(pow_per_ap[u])]]) # 2, 1
                elif self.par.box_mode == 2:
                    u_ap_pairs[u] = torch.vstack((torch.full((1,num_valid_APs_per_u[u]), u), valid_APs[None,:])) # 2, num_valid_APs
                # For bilateration loss:
                x = torch.unsqueeze(pow_per_ap[u], -1)
                v = x[valid_APs] - x[valid_APs].T
                # The AP powers should differ by > M_p for a valid AP pair
                w = torch.where(v > self.par.M_p)
                if len(w[0]) != 0:
                    ap_pairs = torch.vstack((valid_APs[w[0]], valid_APs[w[1]])) # 2, num_valid_AP_pairs
                    u_ap_ap_triplets[u] = torch.vstack((torch.full((1,ap_pairs.shape[1]), u), ap_pairs)) # 3, num_valid_AP_pairs
        
        print('Calculating all anchors and corresponding positive and negative sample intervals...')
        tic = time.time()
        anchors, pos_edges, neg_edges = get_pos_neg_edges(timestamps, self.par.Tc, self.par.Tf, self.par.segment_start_idcs)
        toc = time.time()
        num_anchors = len(anchors)
        print(f'... took {toc - tic: .3f} s for {num_anchors} anchors among {X.shape[0]} samples!')

        # Batchify the indices of _anchors_ (to easily get the pos-neg edges for each anchor)
        # in case not all points are anchors
        all_anchor_idcs = torch.arange(num_anchors)
        train_loader = torch.utils.data.DataLoader(
            all_anchor_idcs, batch_size=self.par.batch_size, shuffle=True)
        num_batches = len(train_loader) 
        
        # Set the module in training mode
        self.network.train()  
        # Visualize training duration
        progress_bar = trange(self.par.num_epochs)  
        # To track the decrease of loss during training:
        loss_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
        # Always track rotation losses so we can diagnose the rotation head.
        loss_rot_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
        loss_align_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
        if plot_each_loss:
            losst_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
            lossbi_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
            lossbox_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)

        # Training loop
        for epoch_idx in progress_bar:
            
            for batch_idx, anchor_idcs in enumerate(train_loader):
                
                if epoch_idx < self.par.epoch_L1_on: # Only bilateration or box loss, so only anchors are forwarded in the network
                    dataset_idcs = anchors[anchor_idcs]
                    batch_x = X[dataset_idcs].to(self.device)
                    # Set all the gradients to zero
                    self.network.zero_grad()
                    # Make a forward pass. The network returns a (pos, rot)
                    # tuple; unpack so the bilateration / box loss code below
                    # can reference batch_y_pos.
                    batch_y = self.network(batch_x)
                    batch_y_pos, batch_y_rot = batch_y[:, :2], batch_y[:, 2:]
                    
                    # Map from the dataset index to the index in the current batch
                    mapper = torch.zeros((np.max(dataset_idcs) + 1), dtype=int, device=self.device)
                    mapper[dataset_idcs] = torch.arange(batch_x.shape[0], dtype=int, device=self.device) 
                
                    loss_position = torch.zeros((), device=self.device) # bc triplet loss = 0
                    loss_rotation = torch.zeros((), device=self.device) # bc triplet loss = 0
                    loss_align = torch.zeros((), device=self.device)   # bc no triplets to form pairs from
                    
                else: # Triplet loss is included!
                    # Get anchor - positive - negative sample triplets
                    triplet_batch = get_triplet_batch(anchors[anchor_idcs], pos_edges[anchor_idcs], neg_edges[anchor_idcs], self.par.num_triplets_per_anchor)
                    # Set all the gradients to zero
                    self.network.zero_grad()
                    # Pass the triplets through the network and compute the loss
                    batch_y_pos, batch_y_rot, mapper, dataset_idcs, loss_position, loss_rotation, loss_align = triplet_pass(self, X, triplet_batch)                                   
                    
                if self.par.lambda_b != 0: # Calculate bilateration loss 
                    
                    # Get the subset of UE-AP-AP triplets to be used in the bilateration loss 
                    cur_u_ap_ap_triplets = torch.cat(list(itemgetter(*dataset_idcs)(u_ap_ap_triplets)), -1).int()
                    # Calculate and add the bilateration loss
                    if cur_u_ap_ap_triplets.shape[1] > 0:
                        cur_u_ap_ap_triplets = cur_u_ap_ap_triplets.to(self.device) # faster
                        loss_b = triplet_margin_loss(batch_y_pos[mapper[cur_u_ap_ap_triplets[0]]], ap_pos[cur_u_ap_ap_triplets[1]], 
                                                      ap_pos[cur_u_ap_ap_triplets[2]], margin=self.par.M_b)
                        
                        loss_position += self.par.lambda_b * loss_b  
                        
                if self.par.lambda_box != 0:
                    if self.par.bb_src == 'ap': # ap-based
                        cur_u_ap_pairs = torch.cat(list(itemgetter(*dataset_idcs)(u_ap_pairs)), -1).int()
                        loss_box = 0
                        if cur_u_ap_pairs.shape[1] > 0:
                            cur_u_ap_pairs = cur_u_ap_pairs.to(self.device) # faster
                            loss_box = bb_loss(batch_y_pos[mapper[cur_u_ap_pairs[0]]], bounding_boxes[cur_u_ap_pairs[1]])
                    else: # genie
                        loss_box = bb_loss(batch_y_pos, bounding_boxes[box_labels[dataset_idcs]])
                        
                    loss_position += self.par.lambda_box * loss_box

                lambda_rotation = getattr(self.par, 'lambda_rotation', 1.0)
                lambda_align = getattr(self.par, 'lambda_align', 0.0)

                loss_total = loss_position + lambda_rotation * loss_rotation + lambda_align * loss_align
                # Backpropagate to get the gradients
                loss_total.backward()
                self.optimizer.step()
                loss_per_epoch[epoch_idx] += loss_total.detach() 
                # Log rotation-related losses for diagnostics
                if torch.is_tensor(loss_rotation):
                    loss_rot_per_epoch[epoch_idx] += loss_rotation.detach()
                if torch.is_tensor(loss_align):
                    loss_align_per_epoch[epoch_idx] += loss_align.detach()

            if self.par.use_scheduler == 1: self.scheduler.step()

        if plot_each_loss:
            losst_per_epoch = losst_per_epoch.cpu().numpy() / num_batches
            lossbi_per_epoch = lossbi_per_epoch.cpu().numpy() / num_batches
            lossbox_per_epoch = lossbox_per_epoch.cpu().numpy() / num_batches
        loss_per_epoch = loss_per_epoch.cpu().numpy() / num_batches
        
        if plot_each_loss:
            plt.figure()
            plt.plot(losst_per_epoch)
            plt.title('loss t')
            
            if self.par.lambda_b != 0:
                plt.figure()
                plt.plot(lossbi_per_epoch)
                plt.title('loss bi')
            if self.par.lambda_box != 0:
                plt.figure()
                plt.plot(lossbox_per_epoch)
                plt.title('loss box')
        
        return loss_per_epoch
    

# %%
class SupervisedModel:
    """Train neural positioning from position and orientation labels.

    The four network outputs are ``(x, z, cos(phi), sin(phi))``. Position uses
    squared Euclidean error. Orientation uses cosine dissimilarity between normalized
    two-dimensional heading vectors.
    """

    def __init__(self, device, par):
        self.device = device
        self.par = par
        self.network = FCNet(self.par.in_features, self.par.hidden_features, self.par.out_features)
        summary(
            self.network,
            input_size=(self.par.batch_size, self.par.in_features),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.par.learning_rate)
        if par.use_scheduler == 1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, par.scheduler_param)
        self.network.to(device)

    @staticmethod
    def angular_distance(predicted, target):
        """Return ``1-cos(delta_phi)`` for pairs of heading vectors."""
        predicted = predicted / (predicted.norm(dim=-1, keepdim=True) + 1e-8)
        target = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
        return 1.0 - (predicted * target).sum(dim=-1)

    @staticmethod
    def angle_targets(angles):
        """Convert raw radians to the legacy ``(cos(phi), sin(phi))`` order."""
        angles = angles.squeeze(-1)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def train(self, dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset):
        """Train and return total, position, rotation, and validation histories."""
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.par.batch_size, shuffle=True
        )
        validation_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.par.batch_size, shuffle=True
        )
        num_batches = len(train_loader)
        total_history = torch.zeros(self.par.num_epochs, device=self.device)
        position_history = torch.zeros(self.par.num_epochs, device=self.device)
        rotation_history = torch.zeros(self.par.num_epochs, device=self.device)
        validation_history = np.zeros(self.par.num_epochs)
        rotation_weight = getattr(self.par, 'lambda_rotation', 1.0)

        progress_bar = trange(self.par.num_epochs)
        for epoch_idx in progress_bar:
            self.network.train()
            total_sum = torch.zeros((), device=self.device)
            position_sum = torch.zeros((), device=self.device)
            rotation_sum = torch.zeros((), device=self.device)

            for batch_x, batch_position, batch_angle in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.to(self.device)
                batch_position = batch_position.to(self.device)
                angle_target = self.angle_targets(batch_angle.to(self.device))
                output = self.network(batch_x)
                predicted_position = output[:, :2]
                predicted_rotation = output[:, 2:]
                position_loss = torch.mean(
                    torch.linalg.norm(predicted_position - batch_position, 2, -1) ** 2
                )
                rotation_loss = torch.mean(
                    self.angular_distance(predicted_rotation, angle_target)
                )
                total_loss = position_loss + rotation_weight * rotation_loss
                total_loss.backward()
                self.optimizer.step()
                total_sum += total_loss.detach()
                position_sum += position_loss.detach()
                rotation_sum += rotation_loss.detach()

            total_history[epoch_idx] = total_sum / num_batches
            position_history[epoch_idx] = position_sum / num_batches
            rotation_history[epoch_idx] = rotation_sum / num_batches
            if self.par.use_scheduler == 1:
                self.scheduler.step()

            self.network.eval()
            validation_sum = 0.0
            with torch.no_grad():
                for batch_x, batch_position, batch_angle in validation_loader:
                    batch_x = batch_x.to(self.device)
                    batch_position = batch_position.to(self.device)
                    angle_target = self.angle_targets(batch_angle.to(self.device))
                    output = self.network(batch_x)
                    position_loss = torch.sum(
                        torch.linalg.norm(output[:, :2] - batch_position, 2, -1) ** 2
                    )
                    rotation_loss = torch.sum(
                        self.angular_distance(output[:, 2:], angle_target)
                    )
                    validation_sum += (
                        position_loss + rotation_weight * rotation_loss
                    ).item()
            validation_history[epoch_idx] = validation_sum / len(
                validation_loader.dataset
            )

        return (
            total_history.cpu().numpy(),
            position_history.cpu().numpy(),
            rotation_history.cpu().numpy(),
            validation_history,
        )

#%%    
class SemisupervisedModel:
    """
    A class to train an FCNet using the triplet loss 
    in addition to an MSE loss with some known ground truth labels.
    This implies semisupervised learning since only a small portion of 
    the training dataset is labeled.
    
    Attributes
    ----------
    device : torch.device
        Device to use for training, e.g. torch.device("cpu") or torch.device("cuda:0").
    par : Parameter
        Simulation scenario and training-related parameters are all stored in this object.
    
    """
    def __init__(self, device, par):
        
        self.device = device
        self.par = par

        
        self.network = FCNet(self.par.in_features, self.par.hidden_features, self.par.out_features)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.par.learning_rate)
        if par.use_scheduler == 1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, par.scheduler_param) 
           
        self.network.to(device)
        
    def print_params(self):
        if self.par.use_scheduler == 0:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t,  
              'use_sch:', self.par.use_scheduler)   
        else:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t,  
              'use_sch:', self.par.use_scheduler, ', sche param:', self.par.scheduler_param)
    
    def train(self, dataset: torch.utils.data.Dataset, ref_idcs, box_labels=None, bounding_boxes=None):
        """
        Train the network based on model parameters.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Training dataset consisting of CSI features and timestamps.
        ref_idcs : arraylike
        The indices of ground truth-labeled samples.
        
        Returns
        -------
        loss_per_epoch : np.array
            Average loss per batch at each epoch for training visualization.

        """
        X = dataset[:][0]
        timestamps = dataset[:][1]
        gt = (dataset[:][2]).to(self.device)
        
        if bounding_boxes is not None: bounding_boxes = bounding_boxes.to(self.device)
        
        print('Calculating all anchors and corresponding positive and negative sample intervals...')
        tic = time.time()
        anchors, pos_edges, neg_edges = get_pos_neg_edges(timestamps, self.par.Tc, self.par.Tf, self.par.segment_start_idcs)
        toc = time.time()
        num_anchors = len(anchors)
        print(f'... took {toc - tic: .3f} s for {num_anchors} anchors among {X.shape[0]} samples!')

        all_anchor_idcs = torch.arange(num_anchors)
        train_loader = torch.utils.data.DataLoader(
            all_anchor_idcs, batch_size=self.par.batch_size, shuffle=True) 
        
        self.network.train()  # set the module in training mode

        num_batches = len(train_loader)

        progress_bar = trange(self.par.num_epochs)  
        loss_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
    
        for epoch_idx in progress_bar:
            sum_loss_per_batch = torch.tensor(0.0, device=self.device)
            for batch_idx, anchor_idcs in enumerate(train_loader):
                # Get anchor - positive - negative sample triplets
                triplet_batch = get_triplet_batch(anchors[anchor_idcs], pos_edges[anchor_idcs], neg_edges[anchor_idcs], self.par.num_triplets_per_anchor)
                # Set all the gradients to zero
                self.network.zero_grad()
                # Pass the triplets through the network and compute the loss
                batch_y, mapper, dataset_idcs, loss = triplet_pass(self, X, triplet_batch)                                   
                
                # Calculate the MSE loss for the anchor points
                arr = np.intersect1d(dataset_idcs, ref_idcs)
                n_arr = len(arr)
                if n_arr != 0:
                    loss_mse = torch.mean(torch.linalg.norm(batch_y[mapper[arr]] - gt[arr], 2, -1)**2)              
                    loss += loss_mse  
                
                # Backpropagate to get the gradients
                loss.backward()
                self.optimizer.step()
                sum_loss_per_batch = sum_loss_per_batch + loss.detach()  # gpu # Should this be detach or item() or.?
                
            loss_per_epoch[epoch_idx] = sum_loss_per_batch / num_batches
            if self.par.use_scheduler == 1: self.scheduler.step()
            
        loss_per_epoch = loss_per_epoch.cpu().numpy()

        return loss_per_epoch
    
