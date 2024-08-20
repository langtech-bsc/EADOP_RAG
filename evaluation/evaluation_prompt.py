from langchain_core.prompts.chat import ChatPromptTemplate

SYSTEM_MESSAGE = "You are a helpful assistant."
SCORING_TEMPLATE_WITH_REFERENCE_1_TO_5 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}"
            '[Ground Truth]\n{reference}\nBegin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale from 1 to 5 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".  .\n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer].',
        ),
    ]
)

SCORING_TEMPLATE_1_TO_5 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the text by the following criteria.[Criteria] {criteria}"
            "[Text] {prediction}."
            "Begin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale from 1 to 5 \
by strictly following this format: '[[rating]]', for example: 'Rating: [[5]]'. "
        ),
    ]
)

SCORING_TEMPLATE_WITH_QUERY_1_TO_5 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}"
            'Begin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response on a scale from 1 to 5 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". \n\n\
[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer].',
        ),
    ]
)

SCORING_TEMPLATE_WITH_CONTEXT_1_TO_5 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "[Instruction]\nPlease act as an impartial judge \
and evaluate the quality of the response provided by an AI \
assistant to the user question displayed below. {criteria}"
            '[Text]\n{context}\nBegin your evaluation \
by providing a short explanation. Be as objective as possible. \
After providing your explanation, you must rate the response either on a scale from 1 to 5 \
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". \
    [The Start of Assistant\'s Answer]\n{prediction}\n\
[The End of Assistant\'s Answer].',
        ),
    ]
)

SCORING_TEMPLATE_WITH_CONTEXT_0_OR_1 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "{criteria}\
            [text1] : {prediction}. \
            [text2] : {context}. "
        ),
    ]
)

SCORING_TEMPLATE_0_OR_1 = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        (
            "human",
            "{criteria}\
            [Text] : {prediction}. "
        ),
    ]
)







