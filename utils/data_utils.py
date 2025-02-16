import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from utils.graph_utils import relationDes

class Words:
    def __init__(self, words_list):
        self.words_list = words_list

    def __len__(self):
        return len(self.words_list) + 1
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.stop:
                if index.stop < len(self.words_list):
                    return self.words_list[index.start:index.stop:index.step]
                else:
                    return self.words_list[index.start::index.step] + ["unknown words"]
            else:
                return self.words_list[index.start::index.step] + ["unknown words"]
        else:
            if index < len(self.words_list):
                return self.words_list[index]
            elif index == len(self.words_list):
                return "unknown words"
            else:
                raise RuntimeError("index out of range.")
    
    def index(self, word):
        if word in self.words_list:
            return self.words_list.index(word)
        else:
            return len(self.words_list)

def le_dataframe_cleaning(df):
    filter_columns = ["WORD1", "WORD2", "AVG_SCORE", "TYPE"]
    column_indice = [df.columns[0].split(" ").index(each) for each in filter_columns]
    df_split = df[df.columns[0]].str.split(" ", expand=True)
    df_split = df_split[[df_split.columns[each] for each in column_indice]]
    df_split.columns = filter_columns
    return df_split
        
def load_le_dataframe(dataset_path, dataset_name="lexical"):
    file_path = '{}/{}/hyperlex_{}_all_{}.csv'
    train_ds = le_dataframe_cleaning(pd.read_csv(file_path.format(dataset_path, dataset_name, "training", dataset_name)))
    val_ds = le_dataframe_cleaning(pd.read_csv(file_path.format(dataset_path, dataset_name, "dev", dataset_name)))
    test_ds = le_dataframe_cleaning(pd.read_csv(file_path.format(dataset_path, dataset_name, "test", dataset_name)))
    return train_ds, val_ds, test_ds

def load_hf_dataset(dataset_path, dataset_name="EVALution"):
    # get the number of words
    if dataset_name == "lexical" or dataset_name == "random":
        train_df, val_df, test_df = load_le_dataframe(dataset_path, dataset_name=dataset_name)
        ds = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df)
        })
        dataset = pd.DataFrame(ds["train"]).drop_duplicates(subset=["WORD1", "WORD2"])
        words = Words(pd.concat([dataset["WORD1"], dataset["WORD2"]]).drop_duplicates().to_list())
        relations = ["To what degree is X a type of Y?"]
        min_degree, max_degree = dataset["AVG_SCORE"].min(), dataset["AVG_SCORE"].max()
        print(f"dataset: {dataset_name} consists of {len(words)} words, degree range is ({min_degree}, {max_degree}), "+  
              f"and {sum([len(ds[each]) for each in ds.keys()])} triples.")
    else:
        ds = load_dataset(path=dataset_path, name=dataset_name)
        dataset = pd.DataFrame(ds["train"]).drop_duplicates(subset=["head", "tail"])
        words = Words(pd.concat([dataset["head"], dataset["tail"]]).drop_duplicates().to_list())
        relations = dataset["relation"].drop_duplicates().to_list()
        print(f"dataset: {dataset_name} consists of {len(words)} words, {len(relations)} relations, and {sum([len(ds[each]) for each in ds.keys()])} triples.")
        print(f"relations:{relations}")
    return words, relations, ds

def load_regression_le(dataset_path, tokenizer, dataset_name="lexical", batch_size=32):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)
    def tokenize_classification(examples):
        text, labels = [], []
        for i in range(0, len(examples["WORD1"])):
            head, tail, label = examples["WORD1"][i], examples["WORD2"][i], float(examples["AVG_SCORE"][i])
            text.append(f"Today, I finally discovered the relation between {head} and {tail}.")
            labels.append(label)
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels}
    le_dataset = ds.map(function=tokenize_classification,
                         batched=True,
                         batch_size=sum([len(ds[each]) for each in ds.keys()]),
                         remove_columns=["WORD1", "WORD2", "AVG_SCORE", "TYPE"])
    le_dataset.set_format("torch")
    train_loader, test_loader = DataLoader(le_dataset["train"], batch_size=batch_size, shuffle=True), DataLoader(le_dataset["test"], batch_size=batch_size)
    val_loader = DataLoader(le_dataset["validation"], batch_size=batch_size) if len(le_dataset)==3 else None
    return train_loader, test_loader, val_loader, words, relations

def load_kg_regression_le(dataset_path, tokenizer, dataset_name="lexical", batch_size=32):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)
    def tokenize_classification(examples):
        text, labels, h, t = [], [], [], []
        for i in range(0, len(examples["WORD1"])):
            head, tail, label = examples["WORD1"][i], examples["WORD2"][i], float(examples["AVG_SCORE"][i])
            graph_head, graph_tail = words.index(head), words.index(tail)
            text.append(f"Today, I finally discovered the relation between {head} and {tail}.")
            labels.append(label)
            h.append(graph_head)
            t.append(graph_tail)
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels, "h": h, "t": t}
    le_dataset = ds.map(function=tokenize_classification,
                         batched=True,
                         batch_size=sum([len(ds[each]) for each in ds.keys()]),
                         remove_columns=["WORD1", "WORD2", "AVG_SCORE", "TYPE"])
    le_dataset.set_format("torch")
    train_loader, test_loader = DataLoader(le_dataset["train"], batch_size=batch_size, shuffle=True), DataLoader(le_dataset["test"], batch_size=batch_size)
    val_loader = DataLoader(le_dataset["validation"], batch_size=batch_size) if len(le_dataset)==3 else None
    return train_loader, test_loader, val_loader, words, relations

def load_language_seq_classification_lrc(dataset_path, tokenizer, dataset_name="EVALution", batch_size=32):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)
    def tokenize_classification(examples):
        text, labels = [], []
        for i in range(0, len(examples["head"])):
            head, tail, label = examples["head"][i], examples["tail"][i], relations.index(examples["relation"][i])
            text.append(f"Today, I finally discovered the relation between {head} and {tail}.")
            labels.append(label)
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels}
    lrc_dataset = ds.map(function=tokenize_classification,
                         batched=True,
                         batch_size=sum([len(ds[each]) for each in ds.keys()]),
                         remove_columns=["head", "tail", "relation"])
    lrc_dataset.set_format("torch")
    train_loader, test_loader = DataLoader(lrc_dataset["train"], batch_size=batch_size, shuffle=True), DataLoader(lrc_dataset["test"], batch_size=batch_size)
    val_loader = DataLoader(lrc_dataset["validation"], batch_size=batch_size) if len(lrc_dataset)==3 else None
    return train_loader, test_loader, val_loader, words, relations


def load_kg_seq_classification_lrc(dataset_path, tokenizer, dataset_name="EVALution", batch_size=32):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)
    def tokenize_classification(examples):
        text, labels, h, t = [], [], [], []
        for i in range(0, len(examples["head"])):
            head, tail, label, graph_head, graph_tail = examples["head"][i], examples["tail"][i], relations.index(examples["relation"][i]), words.index(examples["head"][i]), words.index(examples["tail"][i])
            text.append(f"Today, I finally discovered the relation between {head} and {tail}.")
            labels.append(label)
            h.append(graph_head)
            t.append(graph_tail)
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels, "h": h, "t": t}
    lrc_dataset = ds.map(function=tokenize_classification,
                    batched=True,
                    batch_size=sum([len(ds[each]) for each in ds.keys()]),
                    remove_columns=["head", "tail", "relation"])
    lrc_dataset.set_format("torch")
    train_loader, test_loader = DataLoader(lrc_dataset["train"], batch_size=batch_size, shuffle=True), DataLoader(lrc_dataset["test"], batch_size=batch_size)
    val_loader = DataLoader(lrc_dataset["validation"], batch_size=batch_size) if len(lrc_dataset)==3 else None
    return train_loader, test_loader, val_loader, words, relations

def load_causal_lm_language_lrc(dataset_path, tokenizer, dataset_name="EVALution", batch_size=32, block_size=10):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)

    def tokenize_classification(examples):
        text = []
        relation_verbalization = ", ".join([relations[i] for i in range(0, len(relations) - 1)]) + f", and {relations[-1]}"
        system_prompt = f"You are a linguistics expert. Please give the semantic relationship between the following two words A and B. You can only answer with these few relations: {relation_verbalization}.\nHere are the descriptions of each relation."
        for each in relations:
            system_prompt += f"{each}: {relationDes[each]}\n"
        block_triples = []
        for i in range(0, len(examples["head"])):
            head, tail, rel = examples["head"][i], examples["tail"][i], examples["relation"][i]
            block_triples.append((head, tail, rel))
            if len(block_triples) == block_size:
                user_prompt = "\n".join([f"A: {each[0]}, B: {each[1]}" for each in block_triples])
                answer_prompt = "\n".join(each[2] for each in block_triples)
                messages = [
                    {"role": "system", 
                    "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"Answers:\n{answer_prompt}"}
                ]
                text.append(tokenizer.apply_chat_template(messages, tokenize=False))
                block_triples.clear()
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )

        labels = text["input_ids"].clone()
        labels[torch.isin(text["input_ids"], torch.tensor(tokenizer.all_special_ids))] = -100
        answer_token = tokenizer.convert_tokens_to_ids("Answers")
        indices = torch.where(labels==answer_token)
        assert indices[0].shape[0] == text["input_ids"].shape[0]
        for i in range(0, indices[1].shape[0]):
            labels[i, :indices[1][i]] = -100 
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels}
    
    train_dataset = ds["train"].map(function=tokenize_classification,
                         batched=True,
                         batch_size=len(ds["train"]),
                         remove_columns=["head", "tail", "relation"])
    train_dataset.set_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    def tokenize_classification_test(examples):
        text, index_labels = [], []
        relation_verbalization = ", ".join([relations[i] for i in range(0, len(relations) - 1)]) + f", and {relations[-1]}"
        system_prompt = f"You are a linguistics expert. Please give the semantic relationship between the following two words A and B. You can only answer with these few relations: {relation_verbalization}.\nHere are the descriptions of each relation."
        for each in relations:
            system_prompt += f"{each}: {relationDes[each]}\n"
        block_triples = []
        for i in range(0, len(examples["head"])):
            head, tail, rel = examples["head"][i], examples["tail"][i], examples["relation"][i]
            block_triples.append((head, tail, rel))
            if len(block_triples) == block_size:
                user_prompt = "\n".join([f"A: {each[0]}, B: {each[1]}" for each in block_triples])
                messages = [
                    {"role": "system", 
                    "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"Answers:\n"}
                ]
                text.append(tokenizer.apply_chat_template(messages, tokenize=False))
                index_labels.append([relations.index(each[2]) for each in block_triples])
                block_triples.clear()
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"][:, :-1], "attention_mask": text["attention_mask"][:, :-1], "labels": index_labels}

    test_dataset = ds["test"].map(function=tokenize_classification_test,
                         batched=True,
                         batch_size=len(ds["test"]),
                         remove_columns=["head", "tail", "relation"])
    test_dataset.set_format("torch")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if len(ds) == 3:
        val_dataset = ds["validation"].map(function=tokenize_classification_test,
                         batched=True,
                         batch_size=len(ds["validation"]),
                         remove_columns=["head", "tail", "relation"])
        val_dataset.set_format("torch")
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None
    return train_loader, test_loader, val_loader, words, relations


def load_kg_causal_lm_language_lrc(dataset_path, tokenizer, dataset_name="EVALution", batch_size=32, block_size=1):
    words, relations, ds = load_hf_dataset(dataset_path, dataset_name)

    def tokenize_classification(examples):
        text, h, t, graph_rel = [], [], [], []
        relation_verbalization = ", ".join([relations[i] for i in range(0, len(relations) - 1)]) + f", and {relations[-1]}"
        system_prompt = f"You are a linguistics expert. Please give the semantic relationship between the following two words A and B. You can only answer with these few relations: {relation_verbalization}.\nHere are the descriptions of each relation."
        for each in relations:
            system_prompt += f"{each}: {relationDes[each]}\n"
        block_triples = []
        for i in range(0, len(examples["head"])):
            head, tail, rel, graph_head, graph_tail = examples["head"][i], examples["tail"][i], examples["relation"][i], words.index(examples["head"][i]), words.index(examples["tail"][i])
            g_rel_index = relations.index(rel)
            block_triples.append((head, tail, rel, graph_head, graph_tail, g_rel_index))
            if len(block_triples) == block_size:
                user_prompt = "\n".join([f"A: {each[0]}, B: {each[1]}" for each in block_triples])
                answer_prompt = "\n".join(each[2] for each in block_triples)
                messages = [
                    {"role": "system", 
                    "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"Answers:\n{answer_prompt}"}
                ]
                text.append(tokenizer.apply_chat_template(messages, tokenize=False))
                h.append([each[3] for each in block_triples])
                t.append([each[4] for each in block_triples])
                graph_rel.append([each[5] for each in block_triples])
                block_triples.clear()
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )

        labels = text["input_ids"].clone()
        labels[torch.isin(text["input_ids"], torch.tensor(tokenizer.all_special_ids))] = -100
        answer_token = tokenizer.convert_tokens_to_ids("Answers")
        indices = torch.where(labels==answer_token)
        assert indices[0].shape[0] == text["input_ids"].shape[0]
        for i in range(0, indices[1].shape[0]):
            labels[i, :indices[1][i]] = -100 
        return {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"], "labels": labels, "h": h, "t": t, "graph_label": graph_rel}
    
    train_dataset = ds["train"].map(function=tokenize_classification,
                         batched=True,
                         batch_size=len(ds["train"]),
                         remove_columns=["head", "tail", "relation"])
    train_dataset.set_format("torch")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    def tokenize_classification_test(examples):
        text, index_labels, h, t = [], [], [], []
        relation_verbalization = ", ".join([relations[i] for i in range(0, len(relations) - 1)]) + f", and {relations[-1]}"
        system_prompt = f"You are a linguistics expert. Please give the semantic relationship between the following two words A and B. You can only answer with these few relations: {relation_verbalization}.\nHere are the descriptions of each relation."
        for each in relations:
            system_prompt += f"{each}: {relationDes[each]}\n"
        block_triples = []
        for i in range(0, len(examples["head"])):
            head, tail, rel, graph_head, graph_tail = examples["head"][i], examples["tail"][i], examples["relation"][i], words.index(examples["head"][i]), words.index(examples["tail"][i])
            block_triples.append((head, tail, rel, graph_head, graph_tail))
            if len(block_triples) == block_size:
                user_prompt = "\n".join([f"A: {each[0]}, B: {each[1]}" for each in block_triples])
                messages = [
                    {"role": "system", 
                    "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "system", "content": f"Answers:\n"}
                ]
                text.append(tokenizer.apply_chat_template(messages, tokenize=False))
                h.append([each[3] for each in block_triples])
                t.append([each[4] for each in block_triples])
                index_labels.append([relations.index(each[2]) for each in block_triples])
                block_triples.clear()
        text = tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        return {"input_ids": text["input_ids"][:, :-1], "attention_mask": text["attention_mask"][:, :-1], "labels": index_labels, "h": h, "t": t}

    test_dataset = ds["test"].map(function=tokenize_classification_test,
                         batched=True,
                         batch_size=len(ds["test"]),
                         remove_columns=["head", "tail", "relation"])
    test_dataset.set_format("torch")
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if len(ds) == 3:
        val_dataset = ds["validation"].map(function=tokenize_classification_test,
                         batched=True,
                         batch_size=len(ds["validation"]),
                         remove_columns=["head", "tail", "relation"])
        val_dataset.set_format("torch")
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    else:
        val_loader = None
    return train_loader, test_loader, val_loader, words, relations

