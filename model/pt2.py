import torch

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, 
                pre_seq_len: int=20,
                prefix_hidden_size: int=0,
                hidden_size: int=1024,
                num_hidden_layers: int=24
        ):
        super().__init__()
        self.prefix_projection = True if prefix_hidden_size > 0 else False
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    

class GLMPrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, 
                pre_seq_len: int=20,
                prefix_hidden_size: int=0,
                hidden_size: int=1024,
                num_hidden_layers: int=24
        ):
        super().__init__()
        self.prefix_projection = True if prefix_hidden_size > 0 else False
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    

class PT2PEFTModel(torch.nn.Module):
    def __init__(self, 
                model: torch.nn.Module=None,
                plm_path: str=None,
                pre_seq_len: int=20,
                prefix_hidden_size: int=0
    ):
        super().__init__()
        self.model = model
        self.plm_path = plm_path
        self.pre_seq_len = pre_seq_len

        self.num_kv_heads = model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else model.config.num_attention_heads
        self.out_ft_dim = model.config.hidden_size // model.config.num_attention_heads * self.num_kv_heads
        self.num_hidden_layers = model.config.num_hidden_layers

        if "deberta" in plm_path:
            for param in self.model.deberta.parameters():
                param.requires_grad = False
        elif "roberta" in plm_path:
            for param in self.model.roberta.parameters():
                param.requires_grad = False
        elif "glm" or "llama" in plm_path:
            for param in self.model.model.parameters():
                param.requires_grad = False
        else:
            print("error")
            exit()

        if "glm" in self.plm_path:
            self.prefix_encoder = GLMPrefixEncoder(pre_seq_len, prefix_hidden_size, 
                                            self.out_ft_dim,model.config.num_hidden_layers)
        else:
            self.prefix_encoder = PrefixEncoder(pre_seq_len, prefix_hidden_size, 
                                            self.out_ft_dim,model.config.num_hidden_layers)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None
    ):
        prefix = torch.arange(self.pre_seq_len, device=self.model.device).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        past_key_values = self.prefix_encoder(prefix)

        if "glm" in self.plm_path:
            past_key_values = past_key_values.view(
                input_ids.shape[0],
                self.pre_seq_len,
                self.num_hidden_layers,
                -1
            ).permute([2, 0, 1, 3]).split(1)
            past_key_values = [each.squeeze(0) for each in past_key_values]
        else:
            past_key_values = past_key_values.view(
                input_ids.shape[0],
                self.pre_seq_len,
                self.num_hidden_layers * 2,
                self.num_kv_heads,
                self.out_ft_dim // self.num_kv_heads
            ).permute([2, 0, 3, 1, 4]).split(2)

        batch_size = input_ids.shape[0]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        outputs = self.model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            labels=labels,
            past_key_values=past_key_values
        )

        return outputs

    def generate(self, max_new_tokens, temperature, 
                input_ids,
                attention_mask,
                **kargs):
        prefix = torch.arange(self.pre_seq_len, device=self.model.device).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        past_key_values = self.prefix_encoder(prefix)

        if "glm" in self.plm_path:
            past_key_values = past_key_values.view(
                input_ids.shape[0],
                self.pre_seq_len,
                self.num_hidden_layers,
                -1
            ).permute([2, 0, 1, 3]).split(1)
            past_key_values = [each.squeeze(0) for each in past_key_values]
        else:
            past_key_values = past_key_values.view(
                input_ids.shape[0],
                self.pre_seq_len,
                self.num_hidden_layers * 2,
                self.num_kv_heads,
                self.out_ft_dim // self.num_kv_heads
            ).permute([2, 0, 3, 1, 4]).split(2)

        batch_size = input_ids.shape[0]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        return self.model.generate(max_new_tokens=max_new_tokens, 
                    temperature=temperature, 
                    past_key_values=past_key_values, 
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,)

