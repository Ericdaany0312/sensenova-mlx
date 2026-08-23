from __future__ import annotations


# Exact formal-model generation instruction from the official utils.py.
SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


def build_t2i_query(prompt: str, *, conditional: bool) -> str:
    """Build the byte-exact official ``neo1_0`` non-thinking T2I query."""
    system = (
        f"<|im_start|>system\n{SYSTEM_MESSAGE_FOR_GEN}<|im_end|>\n"
        if conditional
        else ""
    )
    suffix = "<think>\n\n</think>\n\n<img>" if conditional else "<img>"
    return (
        f"{system}<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{suffix}"
    )
