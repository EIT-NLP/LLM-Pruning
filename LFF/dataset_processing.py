import os
import pickle
from datasets import load_from_disk, load_dataset
import copy

def dataset_processing(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_name}"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    # dataset = load_from_disk(args.dataset_path)
    dataset = load_dataset(args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=args.max_length, truncation=True)

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=20,
        remove_columns=dataset["train"].column_names
    )

    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=args.eval_dataset_ratio)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    

    # 保存 tokenized 数据集
    with open(tokenized_dataset_path, 'wb') as f:
        pickle.dump((train_dataset, valid_dataset), f)

    return train_dataset, valid_dataset

def dataset_processing_router(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_name}_router"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        print(f"{tokenized_dataset_path} exists")
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    # dataset = load_from_disk(args.dataset_path)
    dataset = load_dataset(args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=args.max_length, truncation=True)

    ds_train = dataset["train"].shuffle(seed=42).select(range(50000))
    tokenized_dataset = ds_train.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=20,
        remove_columns=dataset["train"].column_names
    )

    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=args.eval_dataset_ratio)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    

    # 保存 tokenized 数据集
    with open(tokenized_dataset_path, 'wb') as f:
        pickle.dump((train_dataset, valid_dataset), f)

    return train_dataset, valid_dataset

def dataset_processing_lw(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_name}_lw"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    # dataset = load_from_disk(args.dataset_path)
    dataset = load_dataset(args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=128, truncation=True)
    
    ds_train = dataset["train"].shuffle(seed=42).select(range(50000))
    tokenized_dataset = ds_train.map(
        tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=20,
        remove_columns=dataset["train"].column_names
    )

    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=0.01)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    
    # 保存 tokenized 数据集
    with open(tokenized_dataset_path, 'wb') as f:
        pickle.dump((train_dataset, valid_dataset), f)

    return train_dataset, valid_dataset

def dataset_processing_llama13(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_path}"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    dataset = load_from_disk(args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], 
                        max_length=args.max_length, 
                        truncation=True, 
                        padding='max_length')  # padding to max_length

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=20,
        remove_columns=dataset["train"].column_names
    )

    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=args.eval_dataset_ratio)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    

    # 保存 tokenized 数据集
    with open(tokenized_dataset_path, 'wb') as f:
        pickle.dump((train_dataset, valid_dataset), f)

    return train_dataset, valid_dataset
