import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    cohen_kappa_score, f1_score
)
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import torch
from typing import Dict, List
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy


# Configuration - All models are 100% open access
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-large"  # Completely open model
    SCORE_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    RETRIEVAL_K = 3
    TEXT_COLUMN = "Responses"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_INPUT_LENGTH = 300  # In words (approx 400 tokens)
    BFI_MAPPINGS = {
        "Extraversion": {
            "items": {
                "High Sociability": "BFI_6",
                "Socially Reserved": "BFI_1"
            },
            "score_col": "GT_Extraversion"
        },
        "Agreeableness": {
            "items": {
                "Trusting of Others": "BFI_2",
                "Critical of Others": "BFI_7"
            },
            "score_col": "GT_Agreeableness"
        },
        "Conscientiousness": {
            "items": {
                "Demonstrates Thoroughness": "BFI_8",
                "Lacks Self-Discipline": "BFI_3"
            },
            "score_col": "GT_Conscientiousness"
        },
        "Neuroticism": {
            "items": {
                "Experiences Anxiety": "BFI_9",
                "High Emotional Stability": "BFI_4"
            },
            "score_col": "GT_Neuroticism"
        },
        "Openness": {
            "items": {
                "Active Imagination": "BFI_10",
                "Low Aesthetic Interest": "BFI_5"
            },
            "score_col": "GT_Openness"
        }
    }


# 1. Initialize LLM with proper device handling
def initialize_llm():
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        Config.LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Create pipeline without explicit device argument
    text_gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        temperature=0.3
    )
    
    return HuggingFacePipeline(pipeline=text_gen_pipeline)

# 2. Knowledge Base Preparation
def create_knowledge_base(df: pd.DataFrame, bfi_mappings: Dict) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Keep well below 512 tokens
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    documents = []
    for _, row in df.iterrows():
        bfi_items = []
        for trait in bfi_mappings.values():
            for item_name, col in trait['items'].items():
                bfi_items.append(f"{item_name}={row[col]}")
        
        # Split long user descriptions
        user_desc = row[Config.TEXT_COLUMN]
        desc_chunks = text_splitter.split_text(user_desc)
        
        # Create multiple documents if description was split
        for chunk in desc_chunks:
            content = f"BFI Scores: {'; '.join(bfi_items)}. User Description: \"{chunk}\""
            metadata = {trait['score_col']: float(row[trait['score_col']]) for trait in bfi_mappings.values()}
            metadata['source'] = f"row_{_}"
            documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

# 3. Vector Store Creation
def create_vector_store(documents: List[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={'device': Config.DEVICE}
    )
    
    # Get texts and metadatas first
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Generate embeddings as numpy array
    embeddings_array = np.array(embeddings.embed_documents(texts), dtype='float32')
    
    # Create FAISS index directly
    index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings_array)),
        embedding=embeddings,
        metadatas=metadatas
    )
    
    return index

# def create_vector_store(documents: List[Document]):
#     embeddings = HuggingFaceEmbeddings(
#         model_name=Config.EMBEDDING_MODEL,
#         model_kwargs={'device': Config.DEVICE}
#     )
#     return FAISS.from_documents(documents, embeddings)

# 4. RAG Chain Factory
def create_rag_chain(trait_name: str, llm, retriever):
    prompt_template = """Predict the {trait_name} score (possible: {scores}) based on:

Context Examples:
{context}

User Data:
{question}

Return ONLY the score in format "X.Y":"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={
            "scores": ", ".join(map(str, Config.SCORE_RANGE)),
            "trait_name": trait_name
        }
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# 5. Prediction Function with Score Validation
def predict_personality(new_user_data: str, rag_chains: Dict[str, RetrievalQA]):
    # Split input if too long
    if len(new_user_data.split()) > 300:  # Rough token estimate
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(new_user_data)
        new_user_data = ". ".join(chunks[:3])  # Use first 3 chunks
        
    predictions = {}
    for trait, chain in rag_chains.items():
        try:
            result = chain.invoke({"query": new_user_data})
            score_text = result['result'].strip()
            score = float(score_text.split()[-1]) if ' ' in score_text else float(score_text)
            predictions[trait] = max(1.0, min(5.0, round(score * 2) / 2))
        except Exception as e:
            print(f"Error predicting {trait}: {str(e)}")
            predictions[trait] = 3.0
    
    return predictions

# 5.5
def predict_and_evaluate(test_df: pd.DataFrame, rag_chains: Dict[str, RetrievalQA], bfi_mappings: Dict):
    """Make predictions and return comprehensive evaluation metrics"""
    # Bin helpers
    def bin_ground_truth(score):
        if score <= 2:
            return "Low"
        elif score == 3:
            return "Moderate"
        else:
            return "High"

    def bin_prediction(score):
        if score < 2.5:
            return "Low"
        elif score <= 3.4:
            return "Moderate"
        else:
            return "High"

    bin_levels = {"Low": 0, "Moderate": 1, "High": 2}
    
    # Storage for results
    predictions = []
    metrics = []
    
    for _, row in test_df.iterrows():
        # Format test case
        bfi_items = []
        for trait in bfi_mappings.values():
            for item_name, col in trait['items'].items():
                bfi_items.append(f"{item_name}={row[col]}")
        
        test_case = f"BFI Scores: {'; '.join(bfi_items)}. User Description: \"{row[Config.TEXT_COLUMN]}\""
        
        # Get prediction
        pred = predict_personality(test_case, rag_chains)
        pred['id'] = _
        predictions.append(pred)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Add ground truth columns
    for trait in bfi_mappings.keys():
        pred_df[f'true_{trait}'] = test_df[bfi_mappings[trait]['score_col']].values
    
    # Calculate metrics for each trait
    results = []
    for trait in bfi_mappings.keys():
        pred_scores = pred_df[trait].astype(float)
        true_scores = pred_df[f'true_{trait}'].astype(float)
        
        # Filter valid scores
        valid = pred_scores.notna() & true_scores.notna()
        pred_scores = pred_scores[valid]
        true_scores = true_scores[valid]
        
        # Regression metrics
        pearson_r, _ = pearsonr(pred_scores, true_scores)
        mae = mean_absolute_error(true_scores, pred_scores)
        rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
        
        # Binned metrics
        binned_pred = pred_scores.apply(bin_prediction)
        binned_true = true_scores.apply(bin_ground_truth)
        
        exact_match = (binned_pred == binned_true).mean()
        off_by_one = (binned_pred.map(bin_levels) - binned_true.map(bin_levels)).abs().le(1).mean()
        kappa = cohen_kappa_score(binned_true, binned_pred, labels=["Low", "Moderate", "High"])
        
        # F1 scores
        f1_macro = f1_score(binned_true, binned_pred, average="macro", labels=["Low", "Moderate", "High"])
        f1_weighted = f1_score(binned_true, binned_pred, average="weighted", labels=["Low", "Moderate", "High"])
        
        results.append({
            "Trait": trait,
            "Pearson_r": round(pearson_r, 3),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "Exact_Match": round(exact_match, 3),
            "Off_By_One_Accuracy": round(off_by_one, 3),
            "Cohen_Kappa": round(kappa, 3),
            "F1_Macro": round(f1_macro, 3),
            "F1_Weighted": round(f1_weighted, 3),
            "N_Samples": len(pred_scores)
        })
    
    metrics_df = pd.DataFrame(results)
    
    return pred_df, metrics_df



# Main Pipeline
def run_pipeline(train_df: pd.DataFrame, bfi_mappings: Dict, test_df: pd.DataFrame = None):
    print("Initializing LLM...")
    llm = initialize_llm()
    
    print("Creating knowledge base...")
    train_docs = create_knowledge_base(train_df, bfi_mappings)
    
    print("Creating vector store...")
    vectorstore = create_vector_store(train_docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
    
    print("Creating RAG chains...")
    rag_chains = {
        trait: create_rag_chain(trait, llm, retriever)
        for trait in bfi_mappings.keys()
    }
    
    if test_df is not None:
        if not isinstance(test_df, pd.DataFrame):
            raise ValueError("test_df must be a pandas DataFrame")
            
        print("\n--- Making Predictions on Test Set ---")
        predictions = []
        for _, row in test_df.iterrows():
            bfi_items = []
            for trait in bfi_mappings.values():
                for item_name, col in trait['items'].items():
                    bfi_items.append(f"{item_name}={row[col]}")
            
            test_case = f"BFI Scores: {'; '.join(bfi_items)}. User Description: \"{row[Config.TEXT_COLUMN]}\""
            
            # Get prediction
            pred = predict_personality(test_case, rag_chains)
            pred['id'] = _
            predictions.append(pred)
        
        # Convert to DataFrame and merge with ground truth
        pred_df = pd.DataFrame(predictions)
        for trait in bfi_mappings.keys():
            pred_df[f'true_{trait}'] = test_df[bfi_mappings[trait]['score_col']].values
        
        # Calculate metrics
        metrics = {}
        for trait in bfi_mappings.keys():
            true_scores = pred_df[f'true_{trait}']
            pred_scores = pred_df[trait]
            
            metrics[trait] = {
                'MAE': mean_absolute_error(true_scores, pred_scores),
                'Accuracy': accuracy_score(
                    np.round(true_scores * 2),
                    np.round(pred_scores * 2)
                ),
                'Exact_Match': accuracy_score(true_scores, pred_scores)
            }
        
        print("\n=== Evaluation Metrics ===")
        for trait, scores in metrics.items():
            print(f"\n{trait}:")
            print(f"  MAE: {scores['MAE']:.3f}")
            print(f"  Accuracy (±0.5): {scores['Accuracy']:.2%}")
            print(f"  Exact Match: {scores['Exact_Match']:.2%}")
        
        return rag_chains, pred_df, metrics
    
    return rag_chains

def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    # Create bins for stratification (5 bins for 1.0-5.0 range)
    mean_scores = df[[trait['score_col'] for trait in Config.BFI_MAPPINGS.values()]].mean(axis=1)
    df['stratify_bin'] = pd.cut(mean_scores, bins=5, labels=False)
    train_df, test_df = train_test_split( df, test_size=test_size, random_state=random_state, stratify=df['stratify_bin'])
    return train_df.drop(columns=['stratify_bin']), test_df.drop(columns=['stratify_bin'])


if __name__ == "__main__":
    df = pd.read_csv ('/users/PGS0218/julina/projects/LoRA_persona/Conversational_personality/Personality_with_profile_518.csv')
    train_df, test_df = create_train_test_split(df, 0.2)
    rag_chains = run_pipeline(train_df, Config.BFI_MAPPINGS)
    
    # Make predictions and evaluate
    pred_df, metrics_df = predict_and_evaluate(test_df, rag_chains, Config.BFI_MAPPINGS)
    
    # Save results
    pred_df.to_csv('/users/PGS0218/julina/projects/LoRA_persona/Conversational_personality/predictions.csv', index=False)
    metrics_df.to_csv('/users/PGS0218/julina/projects/LoRA_persona/Conversational_personality/evaluation_metrics.csv', index=False)
    
    print("\n=== Evaluation Metrics ===")
    print(metrics_df.to_markdown(index=False))
    
    # Optional: Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x="Trait", y="Pearson_r")
    plt.title("Pearson Correlation by Trait")
    plt.tight_layout()
    plt.savefig("/users/PGS0218/julina/projects/LoRA_persona/Conversational_personality/pearson_correlations.png")
    plt.close()

    print("Predictions saved successfully!")
    import gc
    # del embeddings
    torch.cuda.empty_cache()
    gc.collect()