"""Functions to evaluate the RAG pipeline over many models."""

import argparse
from pathlib import Path

import pandas as pd
# from llama_index import ServiceContext

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.evaluation.compute_scores import (
    compute_scores,
    compute_scores_statistics,
)
from private_gpt.evaluation.generate_responses import generate_responses
from private_gpt.evaluation.utils import (
    download_models,
    get_embedding_model_current_setup,
    get_service_evaluators,
    new_columns_names_mappings,
    read_file,
    setup_exists,
    update_llm_chat_service,
    update_settings_file,
    write_file,
)
from private_gpt.settings.settings import Settings
from private_gpt.settings.settings_loader import load_active_settings
from private_gpt.utils.log_config import logger
from llama_index.llms.llama_cpp import LlamaCPP



def compute_setup_evaluation(
    df: pd.DataFrame,
    dict_names_col: dict[str, str],
    current_setup: dict[str, str],
    service_evaluators: dict[str, str | LlamaCPP],
) -> pd.DataFrame:
    """Compute the evaluation of the current setup and store the input data and results.

    The results are the generated answer, the average time it took, the corresponding
    correctness and semantic similariy scores and their summary (mean, median, mode).

    The column title of each computed result contain the name of the llm and embedding
    models from current setup. The format is "{column_name}__{llm}__{embed_model}",
    without the curly brackets. For example:

        "Response_generated__TheBloke/Mistral-7B-Instruct-v0.1-GGUF__BAAI/bge-small-en-v1.5"

    Args:
        service_ctxt_constant (ServiceContext): service context to use to compute the
            scores. It's advised to use the same along the ifferents setups
            evaluations to be able to compare scores in a meaningful way
        df (pd.DataFrame): df to fill with the new scores
        dict_names_col (dict[str, str]): the expected names in df for each Q/A col
        current_setup (dict[str, str]): a dictionary representing the current setup
            with keys : 'llm' and 'embed_model'

    Returns:
        pd.DataFrame: input concatenated with the generated answers of the specified
            setup, the computed scores and their statistics summary.
    """
    # Log the names of models used in current PrivateGPT setup
    logger.info("Computing scores for setup :")
    logger.info("LLM : %s", current_setup["llm"])
    logger.info("Embedding model : %s", current_setup["embed_model"])

    # If the summary column exists in the newly read file
    nb_summary_rows = 4
    if "mean" in df.iloc[:, 0].values:
        # convert to int for csv files, else int indices interpreted as string
        df.iloc[:-nb_summary_rows, 0] = [int(element) for element in df.iloc[:-nb_summary_rows, 0]]  # type: ignore[assignment]
        df.set_index(df.columns[0], inplace=True, drop=True)
        df_QA = df[:-nb_summary_rows].copy()  # df with question/answer data (without summary)
    # If the summary column exists because it was created in current evaluation run
    elif "mean" in df.index:
        df_QA = df[:-nb_summary_rows].copy()
    # If the summary column doesn't exist
    else:
        df_QA = df.copy()

    # Generate and get PrivateGPT responses and store them in df
    df_QA[dict_names_col["col_gen_responses"]], av_time_gen = generate_responses(list(df_QA[dict_names_col["col_queries"]]), "Query Files")

    # Display the time it took to generate the answers
    logger.info(f"---- Average response generation time : {av_time_gen:.2f} seconds ----")

    # Name the scores to avoid spelling errors
    corr = "correctness"
    sem_sim = "semantic_similarity"

    # Compute the evaluation scores
    df_scores = compute_scores(df_QA, dict_names_col, corr, sem_sim, service_evaluators) ## pb là dedans
    df_scores = compute_scores_statistics(df_scores, corr, sem_sim)
    df = pd.concat([df, df_QA[dict_names_col["col_gen_responses"]], df_scores], axis="columns")

    # Add the average computational time for response generation to the results df
    df.loc["average_time [s]", dict_names_col["col_gen_responses"]] = round(av_time_gen, 2)

    # Concat to the name of newly created columns the llm and embedding model names
    column_mappings = new_columns_names_mappings([dict_names_col["col_gen_responses"], corr, sem_sim], current_setup)
    df.rename(columns=column_mappings, inplace=True)

    return df


def evaluate(
    data_dir: Path,
    data_filename: Path,
    llms_to_test_filename: Path = Path("llms_to_eval_template.csv"),
) -> None:
    """Compute the generated responses and scores for each llm in the input file.

    A setup is a combination of llm and embedding models, from the ones given as
    argument.

    The function stores the result in a new file as "resuls/{data_filename}", which is
    of same type of the input and contains all of its data as well as the generated
    answers, the computed scores and their statistics summary.

    For now, one can evaluate the modes "Query Documents" of "LLM Chat".
    "Context Chunks" mode is not implemented for evaluation yet.

    Args:
        data_dir (Path): the directory for results and data file storage
        data_filename (Path): the name of the file with the queries and
            reference answers
        llms_to_test_filename (Path | None, optional): the name of the file containing
            the repos id and files name of llms used in the service to evaluate.
            Defaults to Path("llms_to_eval_template.csv").
    """
    # Get the file's content
    llms_file = PROJECT_ROOT_PATH / "evaluation" / llms_to_test_filename
    df_llms = read_file(llms_file)

    # Define expected names of input files
    dict_names_col = {
        "col_llm_repo_id": "llm_repo_id",
        "col_llm_model_file": "llm_model_file",
        "col_queries": "Query",
        "col_responses": "Response reference",
        "col_gen_responses": "Response_generated",
    }

    # Check that both expected columns are in df
    if (dict_names_col["col_llm_repo_id"] not in df_llms.columns) or (dict_names_col["col_llm_model_file"] not in df_llms.columns):
        logger.error(
            "The given file of llms to evaluate ('%s') must contain two columns named '%s' and '%s'.",
            llms_file,
            dict_names_col["col_llm_repo_id"],
            dict_names_col["col_llm_model_file"],
        )
        raise KeyError

    llms_repo_ids = df_llms[dict_names_col["col_llm_repo_id"]]
    llms_model_files = df_llms[dict_names_col["col_llm_model_file"]]

    # Read the file (checkups, etc) and store it in a new df
    df = read_file(data_dir / data_filename)

    # Check that the two needed data columns are in file
    if (dict_names_col["col_queries"] not in df.columns) or (dict_names_col["col_responses"] not in df.columns):
        logger.error(
            "The given query-answer file must contain two columns named '%s' and '%s'.", dict_names_col["col_queries"], dict_names_col["col_queries"]
        )
        raise KeyError

    # Get the constant service context to be able to compare the evaluation results
    # between the experiments
    service_evaluators = get_service_evaluators()

    # Get the current embedding model from settings
    embed_model = get_embedding_model_current_setup()

    # For each llm compute the evaluation if llm and embedding combination doesn't exist
    for llm_repo_id, llm_model_file in zip(llms_repo_ids, llms_model_files, strict=False):
        current_setup = {"llm": llm_repo_id, "embed_model": embed_model}
        if setup_exists(list(df.columns), current_setup, dict_names_col["col_gen_responses"]):
            logger.info(
                "Setup 'llm: %s, embed_model: %s' already exists in score file. Skipping.", current_setup["llm"], current_setup["embed_model"]
            )
        else:
            update_settings_file(llm_repo_id, llm_model_file)
            download_models()
            new_settings = Settings(**load_active_settings())
            update_llm_chat_service(new_settings)
            df = compute_setup_evaluation(df, dict_names_col, current_setup, service_evaluators)

    # Create the results directory if doesn't exist
    results_dir = data_dir / "results"
    if not results_dir.exists():
        results_dir.mkdir()

    # Write results in new file of same type as input
    write_file(df, results_dir / data_filename.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa_file", type=Path, required=True, help="name of the file containing the queries and expected answers. Supported format : .xlsx, .csv"
    )
    parser.add_argument("--data_folder", type=Path, required=True, help="name of the folder where the data file is stored")
    parser.add_argument(
        "--llms_file",
        type=Path,
        required=False,
        help="name of the file where the llms repo ids and file names are stored. Supported format : .xlsx, .csv",
        default="llms_to_eval_template.csv",
    )

    args = parser.parse_args()
    data_dir = PROJECT_ROOT_PATH / "evaluation" / f"{args.data_folder}"

    evaluate(data_dir, args.qa_file, args.llms_file)
