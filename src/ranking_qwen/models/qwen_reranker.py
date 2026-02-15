"""Qwen3-Reranker model wrapper for training and inference."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ranking_qwen.utils.logger import get_logger

logger = get_logger(__name__)


class QwenReranker:
    """
    Wrapper for Qwen3-Reranker models (0.6B and 4B variants).
    
    This class handles:
    - Model and tokenizer loading
    - Prompt formatting with system instructions
    - Score computation from yes/no token logits
    - Training-specific loss computation
    """
    
    # Qwen reranker template components
    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    )
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
        attn_implementation: str = "flash_attention_2",
        max_length: int = 8192,
        use_flash_attn: bool = True,
    ):
        """
        Initialize the Qwen reranker.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Data type for model weights
            attn_implementation: Attention implementation ('flash_attention_2' or 'eager')
            max_length: Maximum sequence length
            use_flash_attn: Whether to attempt using flash attention
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading Qwen Reranker: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {torch_dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',  # Qwen uses left padding
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with optional flash attention
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': True,
        }
        
        if use_flash_attn and device == "cuda":
            try:
                model_kwargs['attn_implementation'] = attn_implementation
                logger.info(f"  Using {attn_implementation}")
            except Exception as e:
                logger.warning(f"Flash attention not available: {e}")
                logger.info("  Falling back to eager attention")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.to(device)
        
        # Get token IDs for "yes" and "no"
        self.token_true_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.token_false_id = self.tokenizer.convert_tokens_to_ids('no')
        
        logger.info(f"  Token 'yes' ID: {self.token_true_id}")
        logger.info(f"  Token 'no' ID: {self.token_false_id}")
        
        # Pre-compute prefix and suffix tokens
        self._build_template_tokens()
        
        logger.info("Qwen Reranker loaded successfully")
    
    def _build_template_tokens(self):
        """Pre-compute template prefix and suffix tokens."""
        prefix = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        logger.info(f"  Template prefix tokens: {len(self.prefix_tokens)}")
        logger.info(f"  Template suffix tokens: {len(self.suffix_tokens)}")
    
    def format_input(
        self,
        query: str,
        document: str,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ) -> str:
        """
        Format query and document into reranker input prompt.
        
        Args:
            query: Search query
            document: Document text to rank
            instruction: Task instruction
        
        Returns:
            Formatted prompt string
        """
        user_content = (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        
        full_prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        
        return full_prompt
    
    def tokenize_batch(
        self,
        queries: List[str],
        documents: List[str],
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of query-document pairs.
        
        Args:
            queries: List of queries
            documents: List of documents
            instruction: Task instruction
        
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        assert len(queries) == len(documents), "Queries and documents must have same length"
        
        # Format all prompts
        prompts = [
            self.format_input(q, d, instruction)
            for q, d in zip(queries, documents)
        ]
        
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
        }
    
    @torch.no_grad()
    def compute_scores(
        self,
        queries: Union[str, List[str]],
        documents: Union[str, List[str]],
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        batch_size: Optional[int] = None,
    ) -> Union[float, List[float]]:
        """
        Compute relevance scores for query-document pairs.
        
        Scores are computed as sigmoid(logit_yes - logit_no), ranging from 0 to 1.
        
        Args:
            queries: Single query or list of queries
            documents: Single document or list of documents
            instruction: Task instruction
            batch_size: Batch size for processing (if None, process all at once)
        
        Returns:
            Single score or list of scores
        """
        self.model.eval()
        
        # Handle single inputs
        single_input = isinstance(queries, str)
        if single_input:
            queries = [queries]
            documents = [documents]
        
        # Process in batches if specified
        if batch_size is None:
            batch_size = len(queries)
        
        all_scores = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenize_batch(batch_queries, batch_documents, instruction)
            
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Extract yes/no logits
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            
            # Compute probabilities
            scores = torch.sigmoid(true_logits - false_logits)
            all_scores.extend(scores.cpu().numpy().tolist())
        
        return all_scores[0] if single_input else all_scores
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for training.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Binary labels (batch_size,) where 1 = relevant, 0 = irrelevant
        
        Returns:
            Scalar loss tensor
        """
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]  # Last token logits
        
        # Extract yes/no logits
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        
        # Compute probability of "yes"
        probs = torch.sigmoid(true_logits - false_logits)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(probs, labels.float())
        
        return loss
    
    def save_model(self, save_path: str):
        """
        Save model and tokenizer to disk.
        
        Args:
            save_path: Path to save directory
        """
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Model saved successfully")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=self.model.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        logger.info("Checkpoint loaded successfully")
    
    def get_model(self) -> PreTrainedModel:
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer."""
        return self.tokenizer
    
    def parameters(self):
        """Get model parameters for optimizer."""
        return self.model.parameters()
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()


def create_data_collator(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    prefix_tokens: Optional[List[int]] = None,
    suffix_tokens: Optional[List[int]] = None,
):
    """
    Create a data collator for reranker training.
    
    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        prefix_tokens: Pre-computed prefix tokens
        suffix_tokens: Pre-computed suffix tokens
    
    Returns:
        Collator function
    """
    if prefix_tokens is None:
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    
    if suffix_tokens is None:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    def collate_fn(batch):
        """Collate function for DataLoader."""
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for example in batch:
            # Get user content from messages
            user_content = example['messages'][0]['content']
            
            # Encode user content
            user_tokens = tokenizer.encode(user_content, add_special_tokens=False)
            
            # Combine: prefix + user_content + suffix
            full_tokens = prefix_tokens + user_tokens + suffix_tokens
            
            # Truncate from left if needed
            if len(full_tokens) > max_length:
                full_tokens = full_tokens[-max_length:]
            
            # Pad on the left
            padding_length = max_length - len(full_tokens)
            input_ids = [tokenizer.pad_token_id] * padding_length + full_tokens
            attention_mask = [0] * padding_length + [1] * len(full_tokens)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(example['label'])
        
        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
            'labels': torch.tensor(labels_list, dtype=torch.float),
        }
    
    return collate_fn
