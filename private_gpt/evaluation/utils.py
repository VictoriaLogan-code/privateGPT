"""Utils to run the RAG pipeline's evaluation script."""
import os
import subprocess
from pathlib import Path

import pandas as pd
import ruamel.yaml
from llama_index.core.indices import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings as LlamaIndexSettings
from private_gpt.components.llm.prompt_helper import get_prompt_style
from private_gpt.paths import models_cache_path, models_path

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.paths import models_cache_path
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.settings.settings import Settings
from private_gpt.settings.settings_loader import load_settings_from_profile
from private_gpt.utils.log_config import logger
from llama_index.llms.llama_cpp import LlamaCPP

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer 




# def get_service_ctxt_constant(
#     llm_name: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     embed_model_name: str = "local:BAAI/bge-small-en-v1.5",
# ) -> ServiceContext:
#     """Get the Service context object from models used to evaluate the project.

#     It is important to keep constistency between the scores so that they are
#     comparable. For example, the same embedding model to calculate the semantic
#     similarity between the experiments so that the embeddings sensitivity is
#     the same accross the changes of models for privateGPT.

#     Args:
#         llm_name (str, optional): name of the llm model helping us to test our
#             RAG pipeline. Defaults to "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf".
#         embed_model_name (str, optional): name of the embeddings model helping us to
#             test our RAG pipeline. Defaults to "local:BAAI/bge-small-en-v1.5".

#     Returns:
#         ServiceContext: service context create from specifed models
#     """
#     llm = LlamaCPP(
#         model_path=f"{PROJECT_ROOT_PATH / llm_name}",
#         temperature=0.1,
#         # llama2 has a context window of 4096 tokens,
#         # but we set it lower to allow for some wiggle room
#         context_window=3900,
#         generate_kwargs={},
#         # All to GPU
#         model_kwargs={"n_gpu_layers": -1},
#         # transform inputs into Llama2 format
#         messages_to_prompt=messages_to_prompt,
#         completion_to_prompt=completion_to_prompt,
#         verbose=False,
#     )

#     return ServiceContext.from_defaults(llm=llm, embed_model=embed_model_name)

def get_service_evaluators() -> dict[str, LlamaCPP | str]:
    """Get the settings of LLM and embedder used to evaluate the RAG pipeline.
    
    Keep them constant between experiments so that they remain consistent, thus comparable.
    """
    settings_mode = "evaluators"

    # -- Load the values from settings file ---
    logger.info("Loading evaluators settings from settings-%s.yaml file", settings_mode)

    settings_eval = load_settings_from_profile(settings_mode)

    # -- Embedder initialization --
    logger.info("Initializing the evaluator : LLM (%s, in mode=%s)", settings_eval["llamacpp"]["llm_hf_model_file"], settings_eval["llm"]["mode"])
    
    try:
        from llama_index.embeddings.huggingface import (
            HuggingFaceEmbedding,
        )
    except ImportError as e:
        raise ImportError(
            "Local dependencies not found, install with `poetry install --extras embeddings-huggingface`"
        ) from e

    embed_model = HuggingFaceEmbedding(
        model_name= settings_eval["huggingface"]["embedding_hf_model_name"],
        cache_folder=str(models_cache_path),
    )

    # -- LLM initialization --
    logger.info("Initializing the evaluator : LLM (%s, in mode=%s)", settings_eval["llamacpp"]["llm_hf_model_file"], settings_eval["llm"]["mode"])

    if settings_eval["llm"]["tokenizer"]:
        set_global_tokenizer(
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=settings_eval["llm"]["tokenizer"],
                cache_dir=str(models_cache_path),
            )
        )
    
    try:
        from llama_index.llms.llama_cpp import LlamaCPP
    except ImportError as e:
        raise ImportError(
            "Local dependencies not found, install with `poetry install --extras llms-llama-cpp`"
        ) from e


    prompt_style = get_prompt_style(settings_eval["llamacpp"]["prompt_style"])
    settings_eval_kwargs = {
        "tfs_z": settings_eval["llamacpp"]["tfs_z"],  
        "top_k": settings_eval["llamacpp"]["top_k"],  
        "top_p": settings_eval["llamacpp"]["top_p"],  
        "repeat_penalty": settings_eval["llamacpp"]["repeat_penalty"], 
        "n_gpu_layers": -1,
        "offload_kqv": True,
    }
    llm = LlamaCPP(
        model_path=str(models_path / settings_eval["llamacpp"]["llm_hf_model_file"]),
        temperature=settings_eval["llm"]["temperature"],
        max_new_tokens=settings_eval["llm"]["max_new_tokens"],
        context_window=settings_eval["llm"]["context_window"],
        generate_kwargs={},
        callback_manager=LlamaIndexSettings.callback_manager,
        # All to GPU
        model_kwargs=settings_eval_kwargs,
        # transform inputs into Llama2 format
        messages_to_prompt=prompt_style.messages_to_prompt,
        completion_to_prompt=prompt_style.completion_to_prompt,
        verbose=True,
    )
    return {"embed_model": embed_model, "llm": llm}


def setup_exists(names: list[str], current_setup: dict[str, str], base_name: str) -> bool:
    """Determine if a specific setup exists in a given list of names.

    Args:
        names (list[str]): list of names to check for the specific the setup
        current_setup (dict): dictionary representing the current setup with keys :
            'llm' and 'embed_model'
        base_name (str): base name to check the setup into

    Returns:
        bool: True if the setup already exists in the list of names given, False else.
    """
    for col in names:
        if col == f"{base_name}__{current_setup['llm']}__{current_setup['embed_model']}":
            return True
    return False


def new_columns_names_mappings(names: list[str], current_setup: dict[str, str]) -> dict[str, str]:
    """Create a mapping of each name in a list.

    The resulting list contains base_name__llm_name__embed_model_name, all three
    elements separated by two '_', where base_name is the input name (element of
    names list).

    Args:
        names (list[str]): the list of names we want to concat the setup to
        current_setup (dict): {"llm": llm_name, "embed_model": embed_model_name}

    Returns:
        dict[str, str]: the new names generated
    """
    column_mappings = {}
    for name in names:
        column_mappings[name] = f"{name}__{current_setup['llm']}__{current_setup['embed_model']}"

    return column_mappings


def read_file(filepath: Path) -> pd.DataFrame:
    """Read a file and convert it to a dataframe.

    Args:
        filepath (Path): the path of the file to convert

    Raises:
        FileNotFoundError: if the name of file doesn't exist in the current directory
        NotImplementedError: if the file extension is no supported

    Returns:
        pd.DataFrame: file content
    """
    # Check that the file exists
    if not filepath.exists():
        logger.error("File '%s' doesn't exist. Aborting.", filepath)
        raise FileNotFoundError(filepath)

    # Read the file and store its content in a new df
    match filepath.suffix:
        case ".xlsx":
            df = pd.read_excel(filepath)
        case ".csv":
            df = pd.read_csv(filepath)
        case _:
            logger.error("Type of file '%s' is not implemented. Aborting.", filepath.name)
            raise NotImplementedError(filepath)

    return df


def write_file(df: pd.DataFrame, filepath: Path) -> None:
    """Write the content of a dataframe in a given file path.

    Args:
        df (pd.DataFrame): the content to write in the new file
        filepath (Path): the path to store the file at

    Raises:
        NotImplementedError: if the file extension is no supported
    """
    match filepath.suffix:
        case ".xlsx":
            df.to_excel(filepath, index=True)
        case ".csv":
            df.to_csv(filepath, index=True)
        case _:
            logger.error("Type of file '%s' is not implemented. Aborting.", filepath.name)
            raise NotImplementedError(filepath)


def get_embedding_model_current_setup() -> str:
    """Get the embedding model set in the settings.yaml file.

    Returns:
        str: the name of the embedding model
    """
    # Get settings filepath
    settings_filepath = Path(os.environ.get("PGPT_SETTINGS_FOLDER", PROJECT_ROOT_PATH)) / "settings.yaml"

    # Load settings file content
    yaml = ruamel.yaml.YAML()
    with open(settings_filepath) as file:
        data = yaml.load(file)

    return data["huggingface"]["embedding_hf_model_name"]  # type: ignore[no-any-return]


def update_settings_file(
    new_llm_hf_repo_id: str,
    new_llm_hf_model_file: str,
) -> None:
    """Set the new llm in the settings.yaml file.

    Args:
        new_llm_hf_repo_id (str): repo id of new llm
        new_llm_hf_model_file (str): file name of new llm
    """
    settings_filepath = Path(os.environ.get("PGPT_SETTINGS_FOLDER", PROJECT_ROOT_PATH)) / "settings.yaml"

    # Load the settings file content
    yaml = ruamel.yaml.YAML()
    with open(settings_filepath) as file:
        data = yaml.load(file)

    # Modify the models fields
    data["llamacpp"]["llm_hf_repo_id"] = new_llm_hf_repo_id
    data["llamacpp"]["llm_hf_model_file"] = new_llm_hf_model_file

    with open(settings_filepath, "w") as file:
        yaml.dump(data, file)

    logger.info("Settings file updated ('%s')", settings_filepath.name)


def download_models() -> None:
    """Run the setup script to download the models set in the settings.yaml file."""
    # Run the setup script to download the llm
    subprocess.run(["python", f"{PROJECT_ROOT_PATH}/scripts/setup"])


def update_llm_chat_service(new_settings: Settings) -> None:
    """Update the llm of chat service according to the settings object.

    Args:
        new_settings (Settings): the new settings object containing the
            new llm's repo id and filename
    """
    # Get the chat service
    chat_service = global_injector.get(ChatService)

    # Set the new LLM and embedding components # TODO : check if differ or not
    new_llm_component = LLMComponent(new_settings)
    new_embedding_component = EmbeddingComponent(new_settings)

    chat_service.llm_component = new_llm_component
    chat_service.embedding_component = new_embedding_component

    chat_service.index = VectorStoreIndex.from_vector_store(
        chat_service.vector_store_component.vector_store,
        storage_context=chat_service.storage_context,
        llm=new_llm_component.llm,
        embed_model=new_embedding_component.embedding_model,
        show_progress=True,
    )

    logger.info("Chat service has been updated :")
    logger.info("New LLM model filename : '%s'", new_settings.llamacpp.llm_hf_model_file)
