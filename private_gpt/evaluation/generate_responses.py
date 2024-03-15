import time
from collections.abc import Iterable

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.utils.log_config import logger

# Get the chat service
chat_service = global_injector.get(ChatService)


def generate_response(message: str, mode: str) -> str:
    """Compute the assistant's response to a message.

    Args:
        message (str): the user's message, input for the assistant
        mode (str): the mode of execution.
                    Only "Query Documents" is implemented for now.

    Returns:
        str: the assistant's response
    """

    # def yield_deltas(stream: Iterable[ChatResponse | str]) -> Iterable[str]:
    def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
        """Generator function that processes a stream of chat deltas.

        Accumulates and yields the cumulative response at each step.

        Args:
            stream (Iterable[ChatResponse | str]): the stream of chat deltas

        Yields:
            Iterable[str]: the cumulative responses at each step of processing the
                input stream. The function yields a string at each step of processing
                the input stream.
        """
        full_response: str = ""
        stream = completion_gen.response
        for delta in stream:
            if isinstance(delta, str):
                full_response += str(delta)
            elif isinstance(delta, ChatResponse):
                full_response += delta.delta or ""
            yield full_response  # response to query

    # Query with "user: query" format
    chat_message = ChatMessage(content=message, role=MessageRole.USER)

    match mode:
        case "Query Files":
            query_stream = chat_service.stream_chat(
                messages=[chat_message],
                use_context=True,  # to use the vector database to fetch context before answering
                context_filter=None # we want to consider all files in db
            )
            list_yields = list(yield_deltas(query_stream))
            return list_yields[-1]

        case "LLM Chat (no context from files)":
            llm_stream = chat_service.stream_chat(
                # Takes into account last message
                messages=[chat_message],
                use_context=False,
            )
            list_yields = list(yield_deltas(llm_stream))
            return list_yields[-1]

        case _:
            logger.error("Mode '%s' not implemented. Aborting.", mode)
            raise KeyError


def generate_responses(messages: list[str], mode: str) -> tuple[list[str], float]:
    """Generate the assistant's responses to a list of messages.

    Args:
        messages (list[str]): the list of messages we want the responses of
        mode (str): the mode of execution : "Query Documents" or "LLM Chat"
                    are implemented for now

    Returns:
        list[str]: the assistant's responses
        float: the average duration of response generation (in seconds)
    """
    start_generating = time.time()
    generated_responses = [generate_response(message, mode) for message in messages]
    end_generating = time.time()

    total_time = end_generating - start_generating
    mean_time_per_response = total_time / len(messages)

    return generated_responses, mean_time_per_response
