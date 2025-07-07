# hybrid_llm_eval.py

import os
import pickle
import json
import logging
import re
from tqdm import tqdm
import nest_asyncio
import asyncio
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential
)

from llama_index.core import QueryBundle
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.azure_openai import AzureOpenAI   
from ollama import Client

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class HybridConfig:
    """
    Configuration class for the Hybrid LLM Evaluation system.
    
    This class encapsulates all configuration parameters needed for hybrid evaluation,
    including model settings, server configurations, retrieval parameters, and prompts.
    
    Args:
        hybrid_dir (str): Directory path where results will be saved
        ollama_server (str): URL of the Ollama server (e.g., "http://localhost:11434")
        retriever_top_k (int, optional): Number of documents to retrieve initially. Defaults to 25.
        rerank_top_k (int, optional): Number of documents to keep after reranking. Defaults to 5.
        llm_task (dict): Dict specifying provider, model_name, and options for the task LLM
        llm_eval (dict): Dict specifying provider, model_name, and options for the eval LLM
        ... (other legacy args for backward compatibility)
    
    Attributes:
        hybrid_dir (str): Output directory for results
        ollama_server (str): Ollama server URL
        retriever_top_k (int): Initial retrieval count
        rerank_top_k (int): Final reranked document count
        llm_task (dict): Task LLM config
        llm_eval (dict): Eval LLM config
        ... (other attributes)
    
    Example:
        >>> config = HybridConfig(
        ...     hybrid_dir="./results/hybrid_eval/",
        ...     ollama_server="http://10.5.35.200:16228",
        ...     retriever_top_k=30,
        ...     rerank_top_k=10,
        ...     llm_task={"provider": "openai", "model_name": "gpt-4", "api_key": "sk-..."},
        ...     llm_eval={"provider": "ollama", "model_name": "phi4", "base_url": "http://localhost:11434"}
        ... )
    """
    def __init__(
        self,
        hybrid_dir,
        ollama_server,
        retriever_top_k=25,
        rerank_top_k=5,
        llm_task=None,
        llm_eval=None,
        # legacy args for backward compatibility
        ollama_options=None,
        task_llm="phi4",
        evaluate_llm="gemma3:4b-it-qat",
        provider="ollama",
        openai_api_key=None,
        openai_base_url=None,
        openai_deployment=None,
        openai_api_version=None,
    ):
        self.hybrid_dir = hybrid_dir
        self.ollama_server = ollama_server
        self.retriever_top_k = retriever_top_k
        self.rerank_top_k = rerank_top_k
        # Flexible LLM configs
        self.llm_task = llm_task or {
            "provider": provider,
            "model_name": task_llm,
            "base_url": ollama_server,
            "api_key": openai_api_key,
            "deployment_name": openai_deployment,
            "api_version": openai_api_version,
            "ollama_options": ollama_options or {"temperature": 0.7, "num_predict": 1024, "top_p": 0.95},
        }
        self.llm_eval = llm_eval or {
            "provider": provider,
            "model_name": evaluate_llm,
            "base_url": ollama_server,
            "api_key": openai_api_key,
            "deployment_name": openai_deployment,
            "api_version": openai_api_version,
            "ollama_options": ollama_options or {"temperature": 0.7, "num_predict": 1024, "top_p": 0.95},
        }
        self.ollama_options = ollama_options or {
            "temperature": 0.2,
            "num_predict": 1024,
            "top_p": 0.95,
        }
        self.system_prompt = (
            "You are a highly reliable scientific assistant.\n"
            "You must always reason step-by-step, following explicit roles.\n"
            "You must only output a valid JSON object as the final answer.\n"
            "Do not output anything outside the JSON structure."
        )
        self.rag_prompt = PromptTemplate(
            "You are a scientific research assistant. Your task is to evaluate the scientific claim presented in the question using only the information provided in the context.\n\n"
            "Assign a score between 0 and 10, based on the strength of evidence in the context:\n"
            "- 0 = No supporting evidence\n"
            "- 1–4 = Weak or indirect evidence\n"
            "- 5–7 = Moderate or suggestive evidence\n"
            "- 8–10 = Strong and direct evidence\n\n"
            "Use only the information from the context. Do not use prior knowledge. If no evidence is found, give a score of 0.\n"
            "Include citations by listing the filenames of the documents that contain supporting evidence.\n\n"
            "Respond **only** with a valid JSON object. Do **not** include any explanations or text outside the JSON.\n\n"
            "The JSON response format is:\n"
            "{\n"
            "  \"score\": <integer from 0 to 10>,\n"
            "  \"justification\": \"<brief explanation strictly based on the context>\",\n"
            "}\n\n"
            "Context:\n{context_str}\n\n"
            "Question:\n{query_str}\n\n"
            "Answer:"
        )
        self.query_templates = {
            'sepsis_pathogenesis': "is associated with the pathogenesis of sepsis. Score: Based on evidence of the gene's involvement in the biological processes underlying sepsis, including but not limited to its role in the dysregulated host response to infection, organ dysfunction, or sepsis-related complications",
            'sepsis_immune': "is associated with the host immune response in sepsis. Score: Based on evidence of the gene's involvement in the immune response during sepsis, including but not limited to its role in innate or adaptive immunity, inflammation, or immunosuppression",
            'sepsis_organ': "is associated with sepsis-related organ dysfunction. Score: Based on evidence of the gene's involvement in the development or progression of organ dysfunction in sepsis, including but not limited to its role in cardiovascular, respiratory, renal, hepatic, or neurological dysfunction",
            'sepsis_circulating_leukocytes_immune': "is associated with the immune response of circulating leukocytes in sepsis. Score: Based on evidence of the gene's involvement in the immune response of circulating leukocytes during sepsis, including but not limited to its role in leukocyte activation, migration, or function",
            'sepsis_biomarker_clinic': "or its products are currently being used as a biomarker for sepsis in clinical settings. Score: Based on evidence of the gene or its products' application as biomarkers for diagnosis, prognosis, or monitoring of sepsis in clinical settings, with a focus on their validated use and acceptance in medical practice",
            'sepsis_biomarker_blood': "has potential value as a blood transcriptional biomarker for sepsis. Score: Based on evidence supporting the gene's expression patterns in blood cells as reflective of sepsis or its severity, considering both current research findings and potential for future clinical utility",
            'sepsis_drug': "is a known drug target for sepsis treatment. Score: Based on evidence of the gene or its encoded protein serving as a target for therapeutic intervention in sepsis, including approved drugs targeting this gene, compounds in clinical trials, or promising preclinical studies",
            'sepsis_therapeutic': "is therapeutically relevant for managing sepsis or its complications. Score: Based on evidence linking the gene to the management or treatment of sepsis or its associated complications, including its role as a potential target for adjunctive therapies or personalized treatment strategies"
        }

    def get_llm(self, llm_config, role="task"):
        """
        Return an LLM instance based on the provider and model config dict.
        llm_config: dict with keys 'provider', 'model_name', and provider-specific options
        role: 'task' or 'eval' (for logging/debugging)
        """
        provider = llm_config.get("provider", "ollama")
        model_name = llm_config.get("model_name")
        if provider == "ollama":
            from llama_index.llms.ollama import Ollama
            return Ollama(model=model_name, request_timeout=1200.0, base_url=llm_config.get("base_url", self.ollama_server))
        elif provider == "openai":
            from llama_index.llms.openai import OpenAI
            return OpenAI(model=model_name, api_key=llm_config.get("api_key"))
        elif provider == "azure":
            # AzureOpenAI requires 'engine' (deployment name)
            engine = llm_config.get("engine") or llm_config.get("deployment_name") or model_name
            return AzureOpenAI(
                model=model_name,
                engine=engine,
                api_key=llm_config.get("api_key"),
                azure_endpoint=llm_config.get("base_url"),
                deployment_name=engine,
                api_version=llm_config.get("api_version")
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

class HybridLLMEvaluator:
    """
    A hybrid LLM evaluation system that combines naive LLM responses with RAG (Retrieval-Augmented Generation)
    for comprehensive gene evaluation in sepsis research.
    
    This class performs a three-step evaluation process:
    1. Retrieves relevant scientific documents using RAG
    2. Evaluates claims using both naive LLM knowledge and retrieved evidence
    3. Fuses both approaches to produce a final scientific assessment
    
    The evaluator uses different LLM models for task execution and evaluation, supports async processing,
    and includes faithfulness evaluation of RAG responses.
    
    Args:
        config (HybridConfig): Configuration object containing model settings, server URLs, and evaluation parameters
        index: LlamaIndex vector index for document retrieval
        
    Attributes:
        config (HybridConfig): Configuration settings
        index: Document vector index
        client: Ollama client for direct API calls
        task_llm: LLM for main tasks (RAG queries, fusion)
        eval_llm: LLM for evaluation tasks (faithfulness checking)
        retriever: Vector index retriever for document search
        
    Example:
        >>> config = HybridConfig(
        ...     hybrid_dir="./results/",
        ...     ollama_server="http://localhost:11434",
        ...     task_llm="phi4",
        ...     evaluate_llm="gemma3:4b-it-qat"
        ... )
        >>> evaluator = HybridLLMEvaluator(config, vector_index)
        >>> evaluator.run(naive_results_df, max_genes=10)
    """
    def __init__(self, config: HybridConfig, index):
        self.config = config
        self.index = index
        self.client, self.task_llm, self.eval_llm = self._setup_models()
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=config.retriever_top_k)
        os.makedirs(config.hybrid_dir, exist_ok=True)

    def _setup_models(self):
        """
        Initialize and configure LLM models for task execution and evaluation.
        
        Creates separate LLM instances for different purposes:
        - task_llm: Used for RAG queries and fusion reasoning
        - eval_llm: Used for faithfulness evaluation of RAG responses
        
        Returns:
            tuple: (ollama_client, task_llm, eval_llm)
        """
        # Always instantiate Ollama Client for fusion (if needed)
        client = Client(self.config.ollama_server)
        task_llm = self.config.get_llm(self.config.llm_task, role="task")
        eval_llm = self.config.get_llm(self.config.llm_eval, role="eval")
        return client, task_llm, eval_llm

    def _retriever_rerank(self, query, top_k=None, top_rank=None):
        """
        Retrieve and rerank documents based on query relevance.
        
        Performs a two-stage retrieval process:
        1. Vector similarity search to get initial candidates
        2. Cross-encoder reranking for improved relevance scoring
        
        Args:
            query (str): Search query for document retrieval
            top_k (int, optional): Number of documents to retrieve initially. Defaults to config value.
            top_rank (int, optional): Number of documents to return after reranking. Defaults to config value.
            
        Returns:
            List[NodeWithScore]: Reranked nodes with relevance scores
        """
        top_k = top_k or self.config.retriever_top_k
        top_rank = top_rank or self.config.rerank_top_k
        self.retriever.similarity_top_k = top_k
        retrieved_nodes = self.retriever.retrieve(query)
        query_bundle = QueryBundle(query)
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=top_rank
        )
        return reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    async def _evaluate_faithfulness(self, data):
        """
        Evaluate the faithfulness of RAG-generated responses against retrieved contexts.
        
        Uses LlamaIndex's FaithfulnessEvaluator to check if the generated response
        is properly grounded in the retrieved document contexts.
        
        Args:
            data (dict): Dictionary containing query, response, and retrieved nodes
                - query (str): Original query
                - rag_justification (str): Generated response to evaluate
                - retrieved_nodes: List of retrieved document nodes
                
        Returns:
            dict: Evaluation results with Pass/Fail status and reasoning
                - RAG_Evaluation_Result (str): "Pass" or "Fail"
                - RAG_Reasoning (str): Detailed feedback on faithfulness
        """
        faithfulness_evaluator = FaithfulnessEvaluator(llm=self.eval_llm)
        result = await faithfulness_evaluator.aevaluate(
            query=data['query'],
            response=data["rag_justification"],
            contexts=[node.text for node in data["retrieved_nodes"]],
        )
        return {
            "RAG_Evaluation_Result": "Pass" if result.passing else "Fail",
            "RAG_Reasoning": result.feedback,
        }

    async def _rag_query(self, query_str):
        """
        Perform RAG-based query evaluation using retrieved scientific documents.
        
        This method:
        1. Retrieves and reranks relevant documents
        2. Formats retrieved content as context
        3. Generates LLM evaluation based on evidence strength
        4. Evaluates faithfulness of the response
        5. Collects metadata and references
        
        Args:
            query_str (str): Query string for document retrieval and evaluation
            
        Returns:
            dict or None: RAG results containing:
                - query (str): Original query
                - rag_score (int): Evidence strength score (0-10)
                - rag_justification (str): Evidence-based justification
                - context_str (str): Retrieved document contexts
                - retrieved_nodes: Document nodes with metadata
                - retrieval_scores: Relevance scores for retrieved docs
                - unique_references: Set of unique document references
                - RAG_Evaluation_Result: Faithfulness evaluation result
                - RAG_Reasoning: Faithfulness evaluation reasoning
                Returns None if JSON parsing fails
        """
        nodes = self._retriever_rerank(query_str)
        context_str = "\n---\n".join([node.node.text for node in nodes])
        prompt = self.config.rag_prompt.format(context_str=context_str, query_str=query_str)
        llm_response = await self.task_llm.acomplete(prompt, temperature=0.1)
        try:
            parsed = json.loads(llm_response.text.strip().strip("```json").strip("```").strip())
        except Exception as e:
            logger.error(f"RAG JSON decode error: {e}")
            return None
        scores = [node.score for node in nodes]
        refs = set()
        for r in nodes:
            try:
                ref = f"Title: {r.metadata['title']}, YOP: {r.metadata['pub_year']}, pubmed_id: {r.metadata['pubmed_id']}"
            except:
                ref = f"pubmed_id: {r.metadata.get('pubmed_id','')}"
            refs.add(ref)
        rag_results = {
            "query": query_str,
            "rag_score": parsed['score'],
            "rag_justification": parsed['justification'],
            "rag_usage": llm_response.additional_kwargs,
            "context_str": context_str,
            "retrieved_nodes": nodes,
            "retrieval_scores": scores,
            "prompt": prompt,
            "unique_references": refs,
        }
        rag_results.update(await self._evaluate_faithfulness(rag_results))
        return rag_results

    def _build_fusion_prompt(self, query, naive_score, naive_justification, rag_score, rag_justification):
        """
        Build a structured prompt for fusing naive LLM and RAG-based evaluations.
        
        Creates a multi-role prompt that guides the LLM through:
        1. Critically analyzing naive LLM evidence
        2. Evaluating retrieved document evidence  
        3. Making a final arbitrated decision
        
        Args:
            query (str): Original scientific query
            naive_score (int): Score from naive LLM evaluation (0-10)
            naive_justification (str): Justification from naive LLM
            rag_score (int): Score from RAG evaluation (0-10)  
            rag_justification (str): Justification from RAG evaluation
            
        Returns:
            str: Formatted prompt for fusion evaluation
        """
        return f"""
You are an expert scientific assistant.

You must analyze a scientific question using both naive LLM knowledge and retrieved scientific document evidence.

Question:
{query}

=========================================
Role 1: Naive LLM Critic
Think step-by-step:
- Analyze the Naive LLM evidence.
- How strong is the naive prior knowledge?
- Are there any weaknesses, biases, assumptions?
- How reliable would you consider it?

Evidence:
- Score (0-10): {naive_score}
- Justification: {naive_justification}

Write your reasoning before proceeding.

=========================================
Role 2: Retrieved Evidence Analyst
Think step-by-step:
- Analyze the Retrieved Context evidence.
- How specific, strong, or weak is the retrieved evidence?
- Are there conflicting findings?
- How much confidence can we place in it?

Evidence:
- Score (0-10): {rag_score}
- Justification: {rag_justification}

Write your reasoning before proceeding.

=========================================
Role 3: Final Arbiter
Think step-by-step:
- Compare the Naive LLM reasoning and Retrieved Context reasoning.
- If the Retrieved Context is strong and relevant, give it slightly more weight.
- If both sources agree, increase confidence.
- If they conflict, explain the discrepancy carefully.

IMPORTANT: After your reasoning, output ONLY a valid JSON object. Do not include any text, explanations, or commentary after the JSON. The JSON must be the last thing in your response.

{{
  "final_answer": "<Yes/No/Unclear>",
  "final_score": <float between 0.0-10.0>,
  "scientific_explanation": "<your detailed scientific reasoning here, citing sources if possible>"
}}
"""

    async def _process_one(self, k, naive_result_ss):
        """
        Process a single gene-query combination through the complete hybrid evaluation pipeline.
        
        This method orchestrates the full evaluation workflow:
        1. Extracts gene and query information from the dataset
        2. Checks if results already exist (for resumption capability)
        3. Performs RAG-based evaluation using retrieved documents
        4. Builds fusion prompt combining naive and RAG evidence
        5. Generates final hybrid evaluation with retry logic
        6. Saves comprehensive results to pickle file
        
        Args:
            k: Index key for the current row in naive_result_ss
            naive_result_ss (pandas.DataFrame): DataFrame containing naive LLM results
            
        Returns:
            bool: True if processing succeeded, False if failed
            
        The saved results include:
            - All original naive LLM data (score, justification)
            - RAG evaluation results (score, justification, retrieved docs)
            - Fusion evaluation results (final_answer, final_score, scientific_explanation)
            - Metadata (model names, references, faithfulness evaluation)
            - Retrieval information (node scores, usage statistics)
        """
        q_keys = naive_result_ss.loc[k, 'json_key'].replace(" ", "_")
        gene_name = naive_result_ss.loc[k, 'gene_name']
        gene_full_name = naive_result_ss.loc[k, 'gene_full_name']
        outfile = os.path.join(self.config.hybrid_dir, f"{gene_name}_{q_keys}.pkl")
        if os.path.exists(outfile):
            logger.info(f"File {outfile} exists, skipping.")
            return True
        naive_score = naive_result_ss.loc[k, 'score']
        naive_justification = naive_result_ss.loc[k, 'justification']
        query_k = f"The gene symbol {gene_name} or gene name {gene_full_name} {self.config.query_templates[q_keys]}"
        rag_query = query_k.split("Score:")[0]
        rag_result = await self._rag_query(rag_query)
        if rag_result is None:
            logger.error(f"Failed RAG for {gene_name} {q_keys}")
            return False
        f_prompt = self._build_fusion_prompt(
            query=query_k,
            naive_score=naive_score,
            naive_justification=naive_justification,
            rag_score=rag_result['rag_score'],
            rag_justification=rag_result['rag_justification']
        )
        g_response = None
        for _ in range(3):
            try:
                if self.config.llm_task.get("provider") == "ollama":
                    fusion_model = self.config.llm_task.get("model_name")
                    fusion_options = self.config.llm_task.get("ollama_options", self.config.ollama_options)
                    response = self.client.chat(
                        model=fusion_model,
                        options=fusion_options,
                        messages=[
                            {'role': 'system', 'content': self.config.system_prompt},
                            {'role': 'user', 'content': f_prompt}
                        ]
                    )
                    raw_content = response.message['content']
                    logger.debug(f"Raw LLM response for {gene_name} {q_keys}: {raw_content[:500]}...")
                    g_response = self._fix_and_parse_json(raw_content)
                elif self.config.llm_task.get("provider") in ["openai", "azure"] and hasattr(self.task_llm, 'client'):
                    # Use OpenAI/AzureOpenAI direct API with backoff
                    client = self.task_llm.client
                    messages = [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": f_prompt}
                    ]
                    try:
                        completion = await completion_with_backoff(
                            client=client,
                            model=self.config.llm_task.get("engine") or self.config.llm_task.get("model_name"),
                            response_format={"type": "json_object"},
                            temperature=0.1,
                            messages=messages
                        )
                        raw_content = completion.choices[0].message.content
                    except TypeError:
                        completion = await completion_with_backoff(
                            client=client,
                            model=self.config.llm_task.get("engine") or self.config.llm_task.get("model_name"),
                            temperature=0.1,
                            messages=messages
                        )
                        raw_content = completion.choices[0].message.content
                    logger.debug(f"Raw LLM response for {gene_name} {q_keys}: {raw_content[:500]}...")
                    g_response = self._fix_and_parse_json(raw_content)
                else:
                    # Use LlamaIndex LLM wrapper fallback
                    completion_llm = self.task_llm
                    try:
                        completion = await completion_llm.acomplete(
                            f_prompt,
                            temperature=0.1,
                            response_format={"type": "json_object"}
                        )
                    except TypeError:
                        completion = await completion_llm.acomplete(
                            f_prompt,
                            temperature=0.1
                        )
                    raw_content = completion.text
                    logger.debug(f"Raw LLM response for {gene_name} {q_keys}: {raw_content[:500]}...")
                    g_response = self._fix_and_parse_json(raw_content)
                break
            except Exception as e:
                logger.error(f"Fusion JSON error for {gene_name} {q_keys}: {e}")
                if 'raw_content' in locals():
                    logger.error(f"Raw response that failed: {raw_content}")
                continue
        if g_response is None:
            logger.error(f"Failed fusion for {gene_name} {q_keys}")
            return False
        g_response.update({
            "model": self.config.llm_task.get("model_name"),
            "json_key": q_keys,
            "gene_name": gene_name,
            "query": query_k,
            "naive_score": naive_score,
            "naive_justification": naive_result_ss.loc[k, 'justification'],
            "rag_score": rag_result["rag_score"],
            "rag_justification": rag_result["rag_justification"],
            "rag_node_score": rag_result["retrieval_scores"],
            "rag_noderetrived": rag_result["retrieved_nodes"],
            "rag_usage": rag_result["rag_usage"],
            "reference": rag_result["unique_references"],
            "RAG_Evaluation_Result": rag_result.get("RAG_Evaluation_Result", "Not Evaluated"),
            "RAG_Reasoning": rag_result.get("RAG_Reasoning", ""),
            "RAG_evalutator": self.config.llm_eval.get("model_name")
        })
        with open(outfile, "wb") as f:
            pickle.dump(g_response, f)
        logger.info(f"Saved {outfile}")
        return True

    def _fix_and_parse_json(self, raw_response):
        """
        Extract and parse JSON from LLM response, handling common formatting issues.
        
        This method:
        1. Finds the first opening brace in the response
        2. Uses brace counting to find the matching closing brace
        3. Extracts only the JSON portion, ignoring extra text
        4. Cleans common formatting artifacts (```json markers, etc.)
        5. Attempts JSON parsing with detailed error logging
        
        Args:
            raw_response (str): Raw text response from LLM
            
        Returns:
            dict: Parsed JSON object
            
        Raises:
            ValueError: If no JSON-like content is found
            json.JSONDecodeError: If JSON parsing fails after cleaning
        """
        # Find the first opening brace
        start_idx = raw_response.find('{')
        if start_idx == -1:
            raise ValueError("No JSON-like content found.")
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(raw_response)):
            if raw_response[i] == '{':
                brace_count += 1
            elif raw_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        # Extract the JSON portion
        json_candidate = raw_response[start_idx:end_idx + 1].strip()
        
        # Clean up common issues
        json_candidate = json_candidate.replace('```json', '').replace('```', '').strip()
        
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            # Log the problematic JSON for debugging
            logger.error(f"Failed to parse JSON: {json_candidate}")
            logger.error(f"JSON decode error: {e}")
            raise

    def run(self, naive_result_ss, max_genes=None):
        """
        Run the complete hybrid evaluation process on a dataset.
        
        This is the main entry point for batch processing. It creates an async event loop
        and processes each gene-query combination in the dataset.
        
        Args:
            naive_result_ss (pandas.DataFrame): DataFrame containing naive LLM results with columns:
                - gene_name: Gene symbol or identifier
                - json_key: Query type key (maps to query_templates)
                - score: Naive LLM score (0-10)
                - justification: Naive LLM justification text
            max_genes (int, optional): Maximum number of genes to process. If None, processes all.
            
        Note:
            Results are saved as pickle files in the configured hybrid_dir.
            Existing files are automatically skipped to allow for resumption.
        """
        asyncio.run(self._run_async(naive_result_ss, max_genes))

    async def _run_async(self, naive_result_ss, max_genes=None):
        """
        Asynchronously process hybrid evaluations for all genes in the dataset.
        
        Args:
            naive_result_ss (pandas.DataFrame): DataFrame with naive LLM results
            max_genes (int, optional): Maximum number of genes to process
        """
        indices = naive_result_ss.index[:max_genes] if max_genes else naive_result_ss.index
        for k in tqdm(indices, desc="Hybrid LLM Evaluation"):
            await self._process_one(k, naive_result_ss)

@retry(wait=wait_exponential(multiplier=1, min=1, max=512), stop=stop_after_attempt(20))
async def completion_with_backoff(**kwargs):
    # Expects a 'client' kwarg for OpenAI/AzureOpenAI client
    client = kwargs.pop('client')
    return await client.chat.completions.create(**kwargs)