from typing import Union, Optional, Dict, Callable, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch

from .twin_q_model import TwinQModel


# Encoder/decoder variant of deepQLnetwork_RNN.py.
# Differences vs. deepQLnetwork_RNN.py:
#   * a single recurrent layer (rnn_num_layers = 1) of width 32 (rnn_hidden_size = 32),
#   * an MLP **encoder** before the RNN:  Linear(num_inputs->32) -> ReLU -> Linear(32->32),
#   * an MLP **decoder** after the RNN:   Linear(32->32) -> ReLU -> Linear(32->num_actions)
#     (the decoder's final layer is the Q-value output; there is no separate head).
# The RNN core and R2D2 hidden-state plumbing are unchanged, and both modules' `Model`
# subclass the shared twin_q_model.TwinQModel (optimizers / soft updates / save-load),
# so this module's `Model` is a drop-in replacement for deepQLnetwork_RNN.Model.


class RNN(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_layers, rnn_type):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(num_inputs, hidden_size, num_layers)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(num_inputs, hidden_size, num_layers)
        else:
            raise ValueError("Invalid rnn_type input.")

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
                    if isinstance(prev, Dict):
                        state.append([v for v in prev.values()])
                    else:
                        state.append(prev)
            state = list(zip(*state))

            prev_state = [torch.cat(t, dim=1) for t in state]
            if self.rnn_type == "GRU":
                prev_state = prev_state[0]

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
            if self.rnn_type == "LSTM":
                h, c = next_state
                batch_size = h.shape[1]
                next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
                next_state = list(zip(*next_state))
                next_state = [{k: v for k, v in zip(['h', 'c'], item)} for item in next_state]
            elif self.rnn_type == "GRU":
                batch_size = next_state.shape[1]
                next_state = [torch.chunk(next_state, batch_size, dim=1)]
                next_state = list(zip(*next_state))
                next_state = [{k: v for k, v in zip(['h'], item)} for item in next_state]
        else:
            if self.rnn_type == "LSTM":
                next_state = {k: v for k, v in zip(['h', 'c'], next_state)}
            elif self.rnn_type == "GRU":
                next_state = {k: v for k, v in zip(['h'], next_state)}

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
        output, next_state = self.rnn(inputs, prev_state)
        next_state = self._after_forward(next_state, True) # return next_state in list format
        return output, next_state


class Encoder(nn.Module):
    """MLP encoder applied per timestep before the RNN: Linear -> ReLU -> Linear."""
    def __init__(self, num_inputs, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    """MLP decoder applied per timestep after the RNN: Linear -> ReLU -> Linear(num_actions).

    The final layer produces the Q-values, so no separate head is used.
    """
    def __init__(self, rnn_hidden_size, hidden_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(rnn_hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        q_a = self.fc2(x)
        return {'logit': q_a}


class DRQN(nn.Module):
    """
    Overview:
        Encoder/decoder DRQN: an MLP ``encoder`` lifts each observation into a 32-dim
        feature, a single-layer ``rnn`` (LSTM or GRU, width 32) processes the sequence,
        and an MLP ``decoder`` maps the recurrent feature to per-action Q-values. This
        mirrors deepQLnetwork_RNN.DRQN but replaces the stacked RNN + linear head with
        encoder -> 1-layer RNN -> decoder.
    """

    def __init__(
            self,
            num_inputs: int,
            num_actions: int,
            rnn_type: str,
            enc_dec_hidden_size: int = 32,
            rnn_hidden_size: int = 32,
            rnn_num_layers: int = 1,
    ) -> None:
        """
        Overview:
            Initialize the encoder/decoder DRQN according to the corresponding input arguments.
        Arguments:
            - num_inputs (:obj:`int`): Observation dimension.
            - num_actions (:obj:`int`): Number of discrete actions (Q-value output dimension).
            - rnn_type (:obj:`str`): "LSTM" or "GRU".
            - enc_dec_hidden_size (:obj:`int`): Width of the encoder/decoder hidden layers (default 32).
            - rnn_hidden_size (:obj:`int`): Hidden width of the recurrent layer (default 32).
            - rnn_num_layers (:obj:`int`): Number of recurrent layers (default 1).
        """
        super(DRQN, self).__init__()
        self.encoder = Encoder(num_inputs, enc_dec_hidden_size)
        self.rnn = RNN(enc_dec_hidden_size, rnn_hidden_size, rnn_num_layers, rnn_type)
        self.decoder = Decoder(rnn_hidden_size, enc_dec_hidden_size, num_actions)

    def forward(self, inputs: Dict, inference: bool = False, saved_state_timesteps: Optional[list] = None) -> Dict:
        """
        Overview:
            DRQN forward computation graph: encoder -> rnn -> decoder. Inputs an observation
            tensor and previous rnn state to predict q_value.
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
        # encoder -> rnn -> decoder; the difference between inference and training is that
        # inference takes the data with seq_len=1 (T=1).
        # NOTE: in most situations, set inference=True when evaluate and inference=False when training
        if inference:
            x = self.encoder(x)  # (B, enc_dec_hidden_size)
            x = x.unsqueeze(0)  # for rnn input, put the seq_len of x as 1 instead of none.
            # prev_state: DataType: List[Tuple[torch.Tensor]]; Initially, it is a list of None
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)  # to delete the seq_len dim to match decoder input
            x = self.decoder(x)
            x['next_state'] = next_state
            return x
        else:
            # In order to better explain why rnn needs saved_state and which states need to be stored,
            # let's take r2d2 as an example
            # in r2d2,
            # 1) data['burnin_nstep_obs'] = data['obs'][:bs + self._nstep]
            # 2) data['main_obs'] = data['obs'][bs:-self._nstep]
            # 3) data['target_obs'] = data['obs'][bs + self._nstep:]
            # NOTE: (T, B, N) or (T, B, C, H, W)
            # assert len(x.shape) in [3, 5], x.shape

            # need to transpose to match expected dimensions
            x = torch.transpose(x, 0, 1)  # (T, B, N)

            # encode every timestep before the recurrent unroll
            x = self.encoder(x)  # (T, B, enc_dec_hidden_size)

            # NOTE rnn_embedding stores all hidden_state
            rnn_embedding = []
            hidden_state_list = []
            if saved_state_timesteps is not None:
                saved_state = []

            for t in range(x.shape[0]):  # T timesteps
                # NOTE use x[t:t+1] but not x[t] can keep original dimension
                # print('x[t]',x[t:t+1])
                output, prev_state = self.rnn(x[t:t + 1], prev_state)  # output: (1,B, rnn_hidden_size)
                #^^ does prev_state need to be copied when its redefined?
                if saved_state_timesteps is not None and t + 1 in saved_state_timesteps:
                    saved_state.append(prev_state)
                rnn_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                # only keep ht, {list: x.shape[0]{Tensor:(1, batch_size, rnn_hidden_size)}}
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(rnn_embedding, 0)  # (T, B, rnn_hidden_size)
            x = parallel_wrapper(self.decoder)(x)  # (T, B, action_shape)
            # NOTE: x['next_state'] is the hidden state of the last timestep inputted to lstm
            # the last timestep state including the hidden state (h) and the cell state (c)
            # shape: {list: B{dict: 2{Tensor:(1, 1, rnn_hidden_size}}}
            x['next_state'] = prev_state
            # all hidden state h, this returns a tensor of the dim: seq_len*batch_size*rnn_hidden_size
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

        # NOTE: the initial input shape will be (T, B, N),
        #            means encoder or head should process B trajectorys, each trajectory has T timestep,
        #            but T and B dimension can be both treated as batch_size in encoder and head,
        #            i.e., independent and parallel processing,
        #            so here we need such fn to reshape for encoder or head
        x = x.reshape(T * B, *x.shape[2:])
        x = forward_fn(x)
        x = reshape(x)
        return x

    return wrapper


class Model(TwinQModel):
    """Twin recurrent Q-networks (clipped double Q-learning); see ``.twin_q_model.TwinQModel``
        for the shared optimizer / soft-update / save-load machinery.
    """

    def __init__(self, device, num_inputs, num_actions, rnn_type, learning_rate, tau):
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        super().__init__(device,
                         lambda: DRQN(num_inputs, num_actions, rnn_type),
                         learning_rate, tau)
