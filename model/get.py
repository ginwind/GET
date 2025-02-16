import dgl
import torch
from torch import nn
from dgl.nn.functional import edge_softmax
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, ft_dim, num_heads, attn_drop=0.3):
        super(GAT, self).__init__()
        self.ft_dim = ft_dim
        self.num_heads = num_heads
        self.path_encoding = nn.Linear(3 * ft_dim, ft_dim)
        self.attn1 = nn.Linear(ft_dim, num_heads, bias=False)
        self.attn2 = nn.Parameter(torch.empty(size=(1, num_heads, ft_dim // num_heads)))
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)

    def message_passing(self, edges):
        ft = edges.data["eft"] * edges.data["a_value"] # E x num_heads x in_dim
        return {"ft": ft}

    def forward(self, g, nft, eft):
        edge_features = eft
        node_features_start = nft[g.edges()[0]]
        node_features_end = nft[g.edges()[1]]
        
        epaths = self.path_encoding(torch.concat([node_features_start, edge_features, node_features_end], dim=1))\
                                    .view(edge_features.shape[0], self.num_heads, self.ft_dim // self.num_heads)

        a1 = self.attn1(node_features_start) # E x num_heads
        a2 = (epaths * self.attn2).sum(dim=-1) # E x num_heads
        a = (a1 + a2).unsqueeze(dim=-1) # E x num_heads x 1

        a = F.leaky_relu(a)
        # graph message passing. There is no need for dgl.to_simple() due to the necessity of parallel edges.
        g.edata.update({"eft": epaths, "a": a})
        attention = edge_softmax(g, g.edata.pop("a"))
        g.edata["a_value"] = self.attn_drop(attention)
        g.update_all(self.message_passing, dgl.function.sum("ft", "ft")) # N x num_heads x in_dim
        nnft = g.ndata.pop("ft")
        return F.relu(nnft.view(nnft.shape[0], -1) + nft)


class KnowledgeEncoder(nn.Module):
    def __init__(self,
                 gnn_dim: int = 128,
                 num_gnn_layers: int = 3,
                 num_gnn_heads: int = 8,
                 hidden_dim: int = 1024, 
                 plm_num_kv_heads: int = 8,
                 pre_seq_len: int = 10,
                 plm_num_att_heads: int = 32,
                 num_hidden_layers: int = 24,
                 attn_drop: float = 0.3,
                 importance: torch.tensor = None,
                 layer_hop_adj: torch.tensor = None,
                 ):
        super(KnowledgeEncoder, self).__init__()
        self.num_layers = num_gnn_layers
        self.ft_dim = hidden_dim
        self.gnn_dim = gnn_dim
        self.pre_seq_len = pre_seq_len
        self.out_ft_dim = hidden_dim // plm_num_att_heads * plm_num_kv_heads
        self.num_heads = num_gnn_heads
        self.plm_num_kv_heads = plm_num_kv_heads
        self.num_hidden_layers = num_hidden_layers
        self.importances = importance
        self.adj_indices = layer_hop_adj

        self.lamda_list = torch.zeros((num_hidden_layers-1), requires_grad=True)
        self.trans_gnn = nn.ModuleList()
        self.gnn = nn.ModuleList()
        self.trans = nn.ModuleList()
        for _ in range(0, self.num_hidden_layers):
            self.trans_gnn.append(nn.Linear(self.ft_dim, self.gnn_dim))
            gnn = nn.ModuleList()
            for _ in range(0, self.num_layers):
                gnn.append(GAT(self.gnn_dim, num_gnn_heads, attn_drop))
            self.gnn.append(gnn)
            self.trans.append(nn.Sequential(
                nn.Tanh(),
                nn.Linear(self.gnn_dim, 2 * self.out_ft_dim)
            ))

    def forward(self, graph, nft, rft, h, t):
        """
        nft: num_nodes x ft_dim
        eft: num_edges x ft_dim
        word_pairs: batch_size x 2
        """
        prune = False
        if graph.number_of_nodes() > 2000:
            neighbors = torch.unique(
                self.adj_indices[1][torch.nonzero(torch.isin(self.adj_indices[0], torch.concat([h, t]).flatten())).squeeze()],
            )
            if neighbors.shape[0] > 2000:
                neighbors = neighbors[torch.randint(low=0, high=neighbors.shape[0], size=(2000,), device=self.importances.device)]
            graph = dgl.node_subgraph(graph, neighbors)
            _, importance = torch.topk(self.importances[graph.ndata[dgl.NID]], k=self.pre_seq_len)
            prune = True
        else:
            _, importance = torch.topk(self.importances, k=self.pre_seq_len)

        pkv = []
        past_x = 0
        for i in range(0, self.num_hidden_layers):
            if prune:
                input_nft = nft[i][graph.ndata[dgl.NID]]
            else:
                input_nft = nft[i]
            eft = rft[i][graph.edata["label"]]
            x, gnn_eft = self.trans_gnn[i](input_nft), self.trans_gnn[i](eft)
            for j in range(0, self.num_layers):
                x = self.gnn[i][j](graph, x, gnn_eft)
            past_x = x
            if i > 0:
                x = torch.sigmoid(self.lamda_list[i - 1]/0.1) * x + (1 - torch.sigmoid(self.lamda_list[i - 1]/0.1)) * past_x

            prefix_prompt = x[importance].unsqueeze(0).expand(h.shape[0], -1, -1)
            past_key_values = self.trans[i](prefix_prompt) # batch * seq * dim
            past_key_values = past_key_values.view(
                h.shape[0],
                self.pre_seq_len,
                2,
                self.plm_num_kv_heads,
                self.out_ft_dim // self.plm_num_kv_heads
            ).permute([2, 0, 3, 1, 4]) 
            pkv.append(past_key_values)
        return pkv



class GET(torch.nn.Module):
    def __init__(self,
                model: torch.nn.Module,
                plm_path: str,
                graph: dgl.DGLGraph,
                nft: torch.tensor,
                rft: torch.tensor,
                gnn_dim: int=128,
                num_gnn_layers: int=3,
                num_gnn_heads: int=4,
                pre_seq_len: int=10,
                dropout: float=0.,
                nodes_importance: list = None):
        super().__init__()
        self.model = model
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

        self.graph = graph.to(self.model.device)
        self.nft = nft
        self.rft = rft
        self.pre_seq_len = pre_seq_len
        for each in self.nft:
            each.requires_grad = True
        for each in self.rft:
            each.requires_grad = True
        self.hop_adj = dgl.khop_adj(self.graph, 1).to(self.graph.device)
        for i in range(2, num_gnn_layers):
            self.hop_adj += dgl.khop_adj(self.graph, i).to(self.graph.device)
        self.prefix_encoder = KnowledgeEncoder(
            gnn_dim,
            num_gnn_layers,
            num_gnn_heads,
            hidden_dim=model.config.hidden_size,
            plm_num_kv_heads=model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else model.config.num_attention_heads,
            plm_num_att_heads=model.config.num_attention_heads,
            num_hidden_layers=model.config.num_hidden_layers,
            attn_drop=dropout,
            pre_seq_len=pre_seq_len,
            importance=torch.tensor(nodes_importance, device=self.graph.device),
            layer_hop_adj=torch.where(
                self.hop_adj
            )
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        h=None,
        t=None,
        **kargs
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.prefix_encoder(self.graph, self.nft, self.rft, h, t)

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
                h,
                t,
                **kargs):
        batch_size = input_ids.shape[0]
        past_key_values = self.prefix_encoder(self.graph, self.nft, self.rft, h, t)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        return self.model.generate(max_new_tokens=max_new_tokens, 
                    temperature=temperature, 
                    past_key_values=past_key_values, 
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,)