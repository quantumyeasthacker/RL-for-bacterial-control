from typing import Union, Optional, Dict, Callable, List, Tuple
import treetensor.torch as ttorch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np


class LSTM(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers)

    def _before_forward(self, inputs: torch.Tensor, prev_state: Union[None, List[Dict]]) -> torch.Tensor:
        """
        Overview:
            Preprocesses the inputs and previous states before the LSTM `forward` method.
        Arguments:
            - inputs (:obj:`torch.Tensor`): Input vector of the LSTM cell. Shape: [seq_len, batch_size, input_size]
            - prev_state (:obj:`Union[None, List[Dict]]`): Previous state tensor. Shape: [num_directions*num_layers, \
                batch_size, hidden_size]. If None, prv_state will be initialized to all zeros.
        Returns:
            - prev_state (:obj:`torch.Tensor`): Preprocessed previous state for the LSTM batch.
        """
        seq_len, batch_size = inputs.shape[:2]
        # print('seqlen,batchsize',seq_len,batch_size)

        # if batch_size == 4:
        #     print('before_forward', prev_state)

        if prev_state is None:
            num_directions = 1
            zeros = torch.zeros(
                num_directions * self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device
            )
            prev_state = (zeros, zeros)
        elif isinstance(prev_state, list) or isinstance(prev_state, tuple):
            if len(prev_state) != batch_size:
                raise RuntimeError(
                    "prev_state number is not equal to batch_size: {}/{}".format(len(prev_state), batch_size)
                )
            num_directions = 1
            zeros = torch.zeros(
                num_directions * self.num_layers, 1, self.hidden_size, dtype=inputs.dtype, device=inputs.device
            )
            state = []
            for prev in prev_state:
                if prev is None:
                    state.append([zeros, zeros])
                else:
                    if isinstance(prev, (Dict, ttorch.Tensor)):
                        state.append([v for v in prev.values()])
                    else:
                        state.append(prev)
            state = list(zip(*state))

            # print('lstm state',state)

            prev_state = [torch.cat(t, dim=1) for t in state]
        elif isinstance(prev_state, dict):
            prev_state = list(prev_state.values())
        else:
            raise TypeError("not support prev_state type: {}".format(type(prev_state)))

        return prev_state

    def _after_forward(self,
                       next_state: Tuple[torch.Tensor],
                       list_next_state: bool = False) -> Union[List[Dict], Dict[str, torch.Tensor]]:
        """
        Overview:
            Post-processes the next_state after the LSTM `forward` method.
        Arguments:
            - next_state (:obj:`Tuple[torch.Tensor]`): Tuple containing the next state (h, c).
            - list_next_state (:obj:`bool`, optional): Determines the format of the returned next_state. \
                If True, returns next_state in list format. Default is False.
        Returns:
            - next_state(:obj:`Union[List[Dict], Dict[str, torch.Tensor]]`): The post-processed next_state.
        """
        if list_next_state:
            h, c = next_state
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
            next_state = [{k: v for k, v in zip(['h', 'c'], item)} for item in next_state]
        else:
            next_state = {k: v for k, v in zip(['h', 'c'], next_state)}

        # if batch_size == 4:
        #     print('after_forward',next_state)
        return next_state

    def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.BoolTensor:
        """
        Overview:
            Generates a boolean mask for a batch of sequences with differing lengths.
        Arguments:
            - lengths (:obj:`torch.Tensor`): A tensor with the lengths of each sequence. Shape could be (n, 1) or (n).
            - max_len (:obj:`int`, optional): The padding size. If max_len is None, the padding size is the max length of \
                sequences.
        Returns:
            - masks (:obj:`torch.BoolTensor`): A boolean mask tensor. The mask has the same device as lengths.
        """
        if len(lengths.shape) == 1:
            lengths = lengths.unsqueeze(dim=1)
        bz = lengths.numel()
        if max_len is None:
            max_len = lengths.max()
        else:
            max_len = min(max_len, lengths.max())
        return torch.arange(0, max_len).type_as(lengths).repeat(bz, 1).lt(lengths).to(lengths.device)

    def forward(self, inputs, prev_state):
        prev_state = self._before_forward(inputs, prev_state)
        output, next_state = self.lstm(inputs, prev_state)
        next_state = self._after_forward(next_state, True) # return next_state in list format
        return output, next_state


class Head(nn.Module):
    def __init__(self, hidden_size, num_actions):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, inputs):
        q_a = self.fc(inputs)
        return {'logit': q_a}


class DRQN(nn.Module):
    """
    Overview:
        The neural network structure and computation graph of DRQN (DQN + RNN = DRQN) algorithm, which is the most \
        common DQN variant for sequential data and paratially observable environment. The DRQN is composed of three \
        parts: ``encoder``, ``head`` and ``rnn``. The ``encoder`` is used to extract the feature from various \
        observation, the ``rnn`` is used to process the sequential observation and other data, and the ``head`` is \
        used to compute the Q value of each action dimension.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            num_inputs: int,
            num_actions: int,
            lstm_hidden_size: int = 64,
            lstm_num_layers: int = 3
            # head_layer_num: int = 1,
    ) -> None:
        """
        Overview:
            Initialize the DRQN Model according to the corresponding input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network, defaults to None, \
                then it will be set to the last element of ``encoder_hidden_size_list``.
            - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.ReLU()``.
        """
        super(DRQN, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        # obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        self.rnn = LSTM(num_inputs, lstm_hidden_size, lstm_num_layers)
        self.head = Head(lstm_hidden_size, num_actions)

    def forward(self, inputs: Dict, inference: bool = False, saved_state_timesteps: Optional[list] = None) -> Dict:
        """
        Overview:
            DRQN forward computation graph, input observation tensor to predict q_value.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The dict of input data, including observation and previous rnn state.
            - inference: (:obj:'bool'): Whether to enable inference forward mode, if True, we unroll the one timestep \
                transition, otherwise, we unroll the entire sequence transitions.
            - saved_state_timesteps: (:obj:'Optional[list]'): When inference is False, we unroll the sequence \
                transitions, then we would use this list to indicate how to save and return hidden state.
        ArgumentsKeys:
            - obs (:obj:`torch.Tensor`): The raw observation tensor.
            - prev_state (:obj:`list`): The previous rnn state tensor, whose structure depends on ``lstm_type``.
        Returns:
            - outputs (:obj:`Dict`): The output of DRQN's forward, including logit (q_value) and next state.
        ReturnsKeys:
            - logit (:obj:`torch.Tensor`): Discrete Q-value output of each possible action dimension.
            - next_state (:obj:`list`): The next rnn state tensor, whose structure depends on ``lstm_type``.
        Shapes:
            - obs (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
            - logit (:obj:`torch.Tensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
        """

        x, prev_state = inputs['obs'], inputs['prev_state']
        # print('x.shape', x.shape)
        # for both inference and other cases, the network structure is encoder -> rnn network -> head
        # the difference is inference take the data with seq_len=1 (or T = 1)
        # NOTE(rjy): in most situations, set inference=True when evaluate and inference=False when training
        if inference:
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match head network input
            x = self.head(x)
            x['next_state'] = next_state
            return x
        else:
            # In order to better explain why rnn needs saved_state and which states need to be stored,
            # let's take r2d2 as an example
            # in r2d2,
            # 1) data['burnin_nstep_obs'] = data['obs'][:bs + self._nstep]
            # 2) data['main_obs'] = data['obs'][bs:-self._nstep]
            # 3) data['target_obs'] = data['obs'][bs + self._nstep:]
            # NOTE(rjy): (T, B, N) or (T, B, C, H, W)
            # assert len(x.shape) in [3, 5], x.shape

            # need to transpose to match expected dimensions
            x = torch.transpose(x, 0,1)
            # print('x',x.shape)

            # NOTE(rjy) lstm_embedding stores all hidden_state
            lstm_embedding = []
            # TODO(nyz) how to deal with hidden_size key-value
            hidden_state_list = []
            if saved_state_timesteps is not None:
                saved_state = []

            for t in range(x.shape[0]):  # T timesteps
                # NOTE(rjy) use x[t:t+1] but not x[t] can keep original dimension
                # print('x[t]',x[t:t+1])
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, head_hidden_size)
                #^^ does prev_state need to be copied when its redefined?
                if saved_state_timesteps is not None and t + 1 in saved_state_timesteps:
                    saved_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, head_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)  # (T, B, head_hidden_size)
            x = parallel_wrapper(self.head)(x)  # (T, B, action_shape)
            # NOTE(rjy): x['next_state'] is the hidden state of the last timestep inputted to lstm
            # the last timestep state including the hidden state (h) and the cell state (c)
            # shape: {list: B{dict: 2{Tensor:(1, 1, head_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*head_hidden_size
            # This key is used in qtran, the algorithm requires to retain all h_{t} during training
            x['hidden_state'] = torch.cat(hidden_state_list, dim=0)
            if saved_state_timesteps is not None:
                # the selected saved hidden states, including the hidden state (h) and the cell state (c)
                # in r2d2, set 'saved_hidden_state_timesteps=[self._burnin_step, self._burnin_step + self._nstep]',
                # then saved_state will record the hidden_state for main_obs and target_obs to
                # initialize their lstm (h c)
                x['saved_state'] = saved_state
            return x

def parallel_wrapper(forward_fn: Callable) -> Callable:
    """
    Overview:
        Process timestep T and batch_size B at the same time, in other words, treat different timestep data as
        different trajectories in a batch.
    Arguments:
        - forward_fn (:obj:`Callable`): Normal ``nn.Module`` 's forward function.
    Returns:
        - wrapper (:obj:`Callable`): Wrapped function.
    """

    def wrapper(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        T, B = x.shape[:2]

        def reshape(d):
            if isinstance(d, list):
                d = [reshape(t) for t in d]
            elif isinstance(d, dict):
                d = {k: reshape(v) for k, v in d.items()}
            else:
                d = d.reshape(T, B, *d.shape[1:])
            return d

        # NOTE(rjy): the initial input shape will be (T, B, N),
        #            means encoder or head should process B trajectories, each trajectory has T timestep,
        #            but T and B dimension can be both treated as batch_size in encoder and head,
        #            i.e., independent and parallel processing,
        #            so here we need such fn to reshape for encoder or head
        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x

    return wrapper


class Model:
    def __init__(self, device, num_inputs, num_actions):
        self.device = device
        self.q_1 = DRQN(num_inputs, num_actions).to(device)
        self.q_target_1 = DRQN(num_inputs, num_actions).to(device)

        self.q_2 = DRQN(num_inputs, num_actions).to(device)
        self.q_target_2 = DRQN(num_inputs, num_actions).to(device)

        self.q_target_1.eval()
        self.q_target_2.eval()

        self.q_optimizer_1 = Adam(self.q_1.parameters(), lr=1e-4)
        self.q_optimizer_2 = Adam(self.q_2.parameters(), lr=1e-4)

        self._update(self.q_target_1, self.q_1)
        self._update(self.q_target_2, self.q_2)
        self.tau = 0.005
        self.grad_update_num = 0

    def _smaller_weights_last_layer(self, network, scale):
        """Updates the last layer with smaller weights
        Args:
            network: network to update
            scale: amount to scale down weights of last layer
        """
        last_layers = list(network.state_dict().keys())[-2:]
        for layer in last_layers:
            network.state_dict()[layer] /= scale

    def _update(self, target, local):
        """Set the parametrs of target network to be that of local network
        Args:
            target: target network
            local: local network
        """
        target.load_state_dict(local.state_dict())

    def _soft_update(self, target, local):
        """Soft update of parameters in target Networks
        """
        for target_param, param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(target_param.data
                                    * (1.0 - self.tau)
                                    + param.data * self.tau)

    def update_target_nn(self):
        self._soft_update(self.q_target_1, self.q_1)
        self._soft_update(self.q_target_2, self.q_2)

    def save_networks(self, folder_name):
        """
        Save Networks
        """
        torch.save({"model_state_dict": self.q_1.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_1.state_dict()
                    }, folder_name + "q_1")

        torch.save({"model_state_dict": self.q_2.state_dict(),
                    "optimizer_state_dict": self.q_optimizer_2.state_dict()
                    }, folder_name + "q_2")

        torch.save({"model_state_dict": self.q_target_1.state_dict()},
                   folder_name + "q_target_1")

        torch.save({"model_state_dict": self.q_target_2.state_dict()},
                   folder_name + "q_target_2")

    def load_networks(self, folder_name="./"):
        """Loads networks and optimizer state
        Args:
            folder_name: folder from which to load networks from
        """

        q_checkpoint_1 = torch.load(folder_name + "q_1",
                                         map_location=self.device)
        self.q_1.load_state_dict(q_checkpoint_1["model_state_dict"])
        self.q_optimizer_1.load_state_dict(q_checkpoint_1[
            "optimizer_state_dict"])

        q_checkpoint_2 = torch.load(folder_name + "q_2",
                                         map_location=self.device)
        self.q_2.load_state_dict(q_checkpoint_2["model_state_dict"])
        self.q_optimizer_2.load_state_dict(q_checkpoint_2[
            "optimizer_state_dict"])

        q_target_checkpoint_1 = torch.load(folder_name + "q_target_1",
                                                map_location=self.device)
        self.q_target_1.load_state_dict(
            q_target_checkpoint_1["model_state_dict"])

        q_target_checkpoint_2 = torch.load(folder_name + "q_target_2",
                                                map_location=self.device)
        self.q_target_2.load_state_dict(
            q_target_checkpoint_2["model_state_dict"])