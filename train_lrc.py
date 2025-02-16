import torch
import argparse
from transformers import AutoModel, AutoTokenizer, set_seed, LlamaForSequenceClassification
from model.deberta import DebertaForSequenceClassification
from model.debertaV2 import DebertaV2ForSequenceClassification
from model.pt2 import PT2PEFTModel
from model.get import GET
from utils.data_utils import load_language_seq_classification_lrc, load_kg_seq_classification_lrc, load_causal_lm_language_lrc, load_kg_causal_lm_language_lrc
from utils.graph_utils import create_heterogeneous_entity_graph, create_knowledge_embeddings
from utils.utils import Trainer
from peft import LoraConfig, get_peft_model
device = torch.device("cuda:0")

ap = argparse.ArgumentParser(description='LRC learning')
ap.add_argument('--dataset_path', type=str)
ap.add_argument('--dataset', type=str, default="EVALution")
ap.add_argument('--plm_path', type=str)
ap.add_argument('--peft', type=str, default="FT") # FT, LORA, PT2, GET
ap.add_argument('--lora_alpha', type=int, default=16)
ap.add_argument('--lora_dropout', type=float, default=0.1)
ap.add_argument('--lora_r', type=int, default=64)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--epoch", type=int, default=10)
ap.add_argument("--warm_up_rate", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--lr_min", type=float, default=5e-6)
ap.add_argument("--pre_seq_len", type=int, default=20)
ap.add_argument("--prefix_hidden_size", type=int, default=0)
ap.add_argument("--inference_batch", type=int, default=1024)
ap.add_argument("--gnn_dim", type=int, default=128)
ap.add_argument("--num_gnn_layers", type=int, default=3)
ap.add_argument("--num_gnn_heads", type=int, default=4)
ap.add_argument("--attn_drop", type=float, default=0.)
ap.add_argument("--block_size", type=int, default=10)

args = ap.parse_args()
dataset_name = args.dataset
plm_path = args.plm_path
peft = args.peft

for k, v in sorted((vars(args).items())):
    print(k,"=",v)

if __name__ == "__main__":
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(plm_path, trust_remote_code=True)

    # load dataset
    if "llama" in plm_path:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # sequence classification
    if peft == "GET":
        train_loader, test_loader, val_loader, words, relations = load_kg_seq_classification_lrc(dataset_path=args.dataset_path, tokenizer=tokenizer, dataset_name=dataset_name, batch_size=args.batch_size)
    else:
        train_loader, test_loader, val_loader, words, relations = load_language_seq_classification_lrc(dataset_path=args.dataset_path, tokenizer=tokenizer, dataset_name=dataset_name, batch_size=args.batch_size)

    # load plm
    if "deberta_" in plm_path:
        plm = DebertaForSequenceClassification.from_pretrained(plm_path, num_labels=len(relations)).to(device)
    elif "deberta-v2" in plm_path:
        plm = DebertaV2ForSequenceClassification.from_pretrained(plm_path, num_labels=len(relations)).to(device)
    elif "llama" in plm_path:
        plm = LlamaForSequenceClassification.from_pretrained(plm_path, num_labels=len(relations), pad_token_id=tokenizer.pad_token_id).to(device)
    else:
        raise RuntimeError("PLM is not supported.")

    if peft == "LORA": 
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none", 
            task_type="SEQ_CLS"
        )
        model = get_peft_model(plm, peft_config)
    elif peft == "PT2":
        model = PT2PEFTModel(plm, plm_path, args.pre_seq_len, args.prefix_hidden_size)
    elif peft == "GET":
        graph, new_edges, importance = create_heterogeneous_entity_graph(train_loader, words, relations)
        nft, rft = create_knowledge_embeddings(words, relations, plm=plm, tokenizer=tokenizer, device=device, inference_batch=args.inference_batch)
        if new_edges:
            rft = [torch.concat([each, torch.mean(each, dim=0, keepdim=True)], dim=0) for each in rft]
        model = GET(plm, plm_path, graph, nft, rft, 
                            args.gnn_dim, args.num_gnn_layers, args.num_gnn_heads, dropout=args.attn_drop,
                            pre_seq_len=args.pre_seq_len, nodes_importance=importance)
    elif peft == "FT":
        model = plm

    trainer = Trainer(
        relations,
        tokenizer,
        epoch=args.epoch,
        batch_size=args.batch_size,
        warm_up_rate=args.warm_up_rate,
        lr=args.lr,
        lr_min=args.lr_min,
        device=device,
        model=model,
        words=words
    )

    trainer.train(train_loader, test_loader, val_loader)

    
        