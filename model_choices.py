# input: one video's cls tokens, [T, 384], T is the number of frames, each cls token is 384 dimensions long
# output: a dict with video_logit(one number) and frame_logits([T]), one number per frame
import torch.nn as nn
import torch
class model(nn.Module):
    def __init__(self):
        super().__init__()
        
# Working on M1
class two_layers(model):
    def __init__(self, input_dim = 384, hidden_dim=128, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, 1)
        self.dropout= nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.model = nn.Sequential(self.first_layer,
                        self.relu,
                        self.dropout,
                        self.second_layer)
    
    def forward(self, input_tensor):
        # input: [T, 384]
        # output: dictionary with "video_logit" and "frame_logits"
        frame_logits = self.model(input_tensor).squeeze(-1) # converts [T, 1] to [T]
        # we got one score for each frame, so we need to take the mean/max of it to get video score
        if self.pooling == "mean":
            video_logit = frame_logits.mean()
        elif self.pooling == "max":
            # set the video logit to be the maximum score of all frames
            video_logit = frame_logits.max()
        else: 
            raise ValueError(f"unmatched pooling method: {self.pooling}")    
        return {"video_logit": video_logit, "frame_logits": frame_logits}

class ABMIL(model):
    # attention based multiple instance learning
    def __init__(self, input_dim = 384, hidden_dim = 128):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, 64)
        self.third_layer = nn.Linear(64, 1)
        self.fourth_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.dropout= nn.Dropout(0.2)
        
        # not the type of encoder we learnt in lecture
        self.encoder = nn.Sequential(self.first_layer,
                                   self.relu,
                                   self.dropout)
        # not self attention we learnt in lecture
        self.attention_network = nn.Sequential(self.second_layer,
                                           self.tanh,
                                           self.third_layer,
                                           self.softmax)
    def forward(self, input_tensor):
        # self.model(input_tensor)
        # self.weighted_average = torch.dot_product(self.encoder, self.attention_netowrk)
        # self.model = self.fourth_layer(self.weighted_average)
        #   [T, 128]
        h = self.encoder(input_tensor) 
        #    [T, 1]
        a = self.attention_network(h)
        #    [T, 128]
        z = (a * h)
        #    [128], we add the value on the same dimension across frame, 
        #  because it is how it works, we want to compare each feature at the end
        weighted_average = z.sum(dim = 0)
        video_logit = self.fourth_layer(weighted_average).squeeze(-1)
        frame_logits = self.fourth_layer(h).squeeze(-1)
        return {"video_logit": video_logit, "frame_logits": frame_logits}

    
       