"""Functions to compute the scores of the RAG pipeline."""

# import asyncio

import pandas as pd
from llama_index.core.base.embeddings.base import SimilarityMode
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    EvaluationResult,
    SemanticSimilarityEvaluator,
)

from private_gpt.utils.log_config import logger
from llama_index.llms.llama_cpp import LlamaCPP



def compute_correctness(
    query: str,
    response: str,
    reference: str,
    evaluators_llm: LlamaCPP,
    corr_threshold: float = 4.0,
) -> EvaluationResult:
    """Compute the correctness of given query, answer and expected answer.

    Th correctness score works with prompt engineering asking to give a score
    between 1 (worst) and 5 (best), judging the relevance and correctness of
    an answer given a query and the generated answer.

    Args:
        query (str): the query to compute the score with
        response (str): the response to compute the score with
        reference (str): the reference of response (expected) to compute the score with
        corr_threshold (float, optional): the threshold for passing the correctness
            evaluation. Defaults to 4.0.

    Returns:
        EvaluationResult: an object containing the computed correctness details.
            Its attributes include : passing (bool), feedback (str, the reasoning
            that led to this score) and score (float).

    """
    correctness_evaluator = CorrectnessEvaluator(
        llm=evaluators_llm, # Constant llm to compare the setups
        score_threshold=corr_threshold,
    )

    result_correctness = correctness_evaluator.evaluate(
        query=query,
        response=response,
        reference=reference,
    )

    return result_correctness


def compute_semantic_similarity(
    response: str,
    reference: str,
    evaluators_embed_model: str,
    sem_sim_threshold: float = 0.8,
    sem_sim_mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> EvaluationResult:
    """Compute the semantic similarity between an answer and expected answer.

    Semantic similarity evaluates the quality of a question answering system by
    comparing the similarity between embeddings of the generated answer and the
    reference answer.

    Args:
        response (str): the response to compute the score with
        reference (str): the reference of response (expected) to compute the score with
        sem_sim_threshold (float, optional): the threshold for passing the semantic
            similarity evaluation. Defaults to 0.8.
        sem_sim_mode (SimilarityMode, optional): the semantic similarity mode.
            The options are : .DEFAULT (cosine similarity), .DOT_PRODUCT, .EUCLIDEAN
            and custom function. Defaults to SimilarityMode.DEFAULT.

    Returns:
        EvaluationResult: an object containing the computed semantic similarity details.
            Its attributes include : passing (bool) and score (float).
    """
    semantic_sim_evaluator = SemanticSimilarityEvaluator(
        embed_model=evaluators_embed_model, # Constant embedding model to compare the setups
        similarity_mode=sem_sim_mode,
        similarity_threshold=sem_sim_threshold,
    )

    result_sem_sim = semantic_sim_evaluator.evaluate(
        response=response,
        reference=reference,
    )

    return result_sem_sim


def compute_scores(
    df: pd.DataFrame,
    dict_names_col: dict[str, str],
    corr: str,
    sem_sim: str,
    service_evaluators: dict[str, str | LlamaCPP],
    corr_threshold: float = 4.0,
    sem_sim_threshold: float = 0.8,
    sem_sim_mode: SimilarityMode = SimilarityMode.DEFAULT,
    round_val: int = 2,
) -> pd.DataFrame:
    """Compute correctness and semantic similarity scores of given dataframe.

    The dataframe must have a column with queries, another one with reference answers
    and an other one with generated answers (to compute the score of).

    Args:
        df (pd.DataFrame): the dataframe containing the data to compute the scores of
        dict_names_col (dict[str, str]): the expected names in df for each Q/A col
        corr (str): the name of the correctness score, for consistency across functions
        sem_sim (str): the name of the semantic similarity score, for the same reason
        llm_evaluator (str): the name of the llm to use to compute the scores.
            It is advised to use the same along the ifferents setups evaluations,
            to be able to compare scores in a meaningful way
        embed_evaluator (str): the name of the embedder to use to compute the scores.
            It is advised to use the same along the ifferents setups evaluations,
            to be able to compare scores in a meaningful way
        corr_threshold (float, optional): the threshold for passing the correctness
            evaluation. Defaults to 4.0.
        sem_sim_threshold (int, optional): the threshold for passing the semantic
            similarity evaluation. Defaults to 0.8.
        sem_sim_mode (SimilarityMode, optional): the semantic similarity mode.
            The options are : .DEFAULT (cosine similarity), .DOT_PRODUCT, .EUCLIDEAN
            and custom function. Defaults to SimilarityMode.DEFAULT.
        round_val (int, optional): number of decimals to round to. Defaults to 2.

    Returns:
        pd.DataFrame: the scores computed (same number of rows as input)
    """
    # Get the data to use
    queries = df[dict_names_col["col_queries"]]
    responses_generated = df[dict_names_col["col_responses"]]
    responses_ref = df[dict_names_col["col_gen_responses"]]

    # Create a dictionary to store the scores
    scores: dict[str, list[float]] = {key: [] for key in [corr, sem_sim]}

    df_scores = pd.DataFrame()

    for idx, (query, response, reference) in enumerate(zip(queries, responses_generated, responses_ref, strict=True)):

        # --- CORRECTNESS ---
        result_corr = compute_correctness(query, response, reference, service_evaluators["llm"], corr_threshold) # pb ici (2)
        if result_corr is None:
            logger.error("Row %d of input file : correctness score is None, check your input file, it may contain None values. Aborting.", idx)
            raise ValueError("Input file may contain None values.")
        scores[corr].append(result_corr.score)  # type: ignore[arg-type]

        # --- SEMANTIC SIMILARITY ---
        result_sem_sim = compute_semantic_similarity(response, reference, service_evaluators["embed_model"], sem_sim_threshold, sem_sim_mode)
        if result_sem_sim is None:
            logger.error(
                "Row %d of input file : semantic similarity score is None, check your input file, it may contain None values. Aborting.", idx
            )
            raise ValueError("Input file may contain None values.")
        scores[sem_sim].append(round(result_sem_sim.score, round_val))  # type: ignore[arg-type]

    # Append scores columns to the scores df
    df_scores[corr] = scores[corr]
    df_scores[sem_sim] = scores[sem_sim]

    return df_scores


def compute_scores_statistics(df_scores: pd.DataFrame, corr: str, sem_sim: str) -> pd.DataFrame:
    """Add the statistics of the dataframe of scores given as input.

    The two expected scores are correctness and semantic similarity and the statistics
    computed are mean, median and mode.
    The mode is calculated only for the correctness score, as it only makes sense for
    discrete values.

    Args:
        df_scores (pd.DataFrame): the dataframe containing the scores
        corr (str): the name of the correctness score, for consistency across functions
        sem_sim (str): the name of the semantic_similarity score, for the same reason

    Returns:
        pd.DataFrame: dataframe containing the scores and their statistics
    """
    # Extract columns from the DataFrame
    corr_scores = df_scores[corr]
    semsim_scores = df_scores[sem_sim]

    # Compute summary statistics
    mean_corr = round(corr_scores.mean(), 2)
    median_corr = round(corr_scores.median(), 2)
    mode_corr = corr_scores.mode().iloc[0]  # Mode can be multiple values, take the first

    mean_semsim = round(semsim_scores.mean(), 2)
    median_semsim = round(semsim_scores.median(), 2)

    # Create a summary DataFrame
    df_summary_stats = pd.DataFrame(
        {corr: [mean_corr, median_corr, mode_corr], sem_sim: [mean_semsim, median_semsim, None]}, index=["mean", "median", "mode"]
    )

    # Concat input df (scores) with summary df
    df_scores = pd.concat([df_scores, df_summary_stats], axis="index")

    return df_scores
