conciseness_criteria = {
    "conciseness": """
Score 1: The answer contains too much information and includes unnecessary details that could confuse the reader or detract from the main point.
Score 2: The answer is overly long-winded and could be condensed to convey the same information more efficiently.
Score 3: The answer is generally brief and to the point, but there are still some words or phrases that could be removed to improve clarity and brevity.
Score 4: The answer is clear, succinct, and effectively communicates the necessary information without unnecessary elaboration.
Score 5: The answer is exceptionally brief, yet it effectively conveys all the required information with clarity and precision."""
}

relevance_criteria = {
    "relevance": """
Score 1: The answer is not relevant to the question and does not address the topic at hand.
Score 2: The answer is somewhat relevant but lacks focus and may include tangential information.
Score 3: The answer is relevant but could be more focused on addressing the specific question or topic.
Score 4: The answer is highly relevant and directly addresses the question or topic with appropriate detail.
Score 5: The answer is perfectly relevant, providing precise and comprehensive information directly related to the question or topic."""}

correctness_criteria = {
    "correctness": """
Score 1: The answer contains significant inaccuracies or errors that mislead the reader.
Score 2: The answer has several inaccuracies or errors that affect its credibility and may confuse the reader.
Score 3: The answer is generally correct but contains minor inaccuracies or errors that could be corrected for improved accuracy.
Score 4: The answer is mostly correct, with few inaccuracies or errors that do not significantly impact its overall correctness.
Score 5: The answer is entirely correct, providing accurate and reliable information without any inaccuracies or errors."""
}

understandability_criteria = {
    "understandability": """
Score 1: The answer is overly complex and difficult to understand, using technical jargon or convoluted language that may confuse the reader.
Score 2: The answer is somewhat complex, with unnecessary complexity in language or expression that could be simplified for better understanding.
Score 3: The answer is clear but could be simplified in language or expression to improve accessibility and readability.
Score 4: The answer is straightforward and easy to understand, with language that is clear and accessible to most readers.
Score 5: The answer is exceptionally simple and easy to understand, using plain language and straightforward expression that ensures clarity for all readers."""
}

groundedness_criteria = {
    "groundedness": """
Score 1: The assistant's answer contains facts which are not explicitely mentioned in the provided text.
Score 3: The assistant's answer is mostly grounded in the provided text, using relevant details to support its claims. However, it may contain some minor inaccuracies or areas where the connection to the text could be stronger.
Score 5: The assistant's answer is fully grounded in the provided text, accurately referencing specific details and evidence to support its claims."""
}

completeness_criteria = {
    "completeness": """
Score 1: The answer is significantly incomplete, missing key information and failing to address the main points of the question.
Score 2: The answer is incomplete, omitting several important details and partially addressing the question.
Score 3: The answer is generally complete but lacks some details or explanations that would provide a fuller understanding of the topic.
Score 4: The answer is mostly complete, with only minor details or explanations missing that do not significantly impact the overall understanding.
Score 5: The answer is entirely complete, thoroughly addressing all aspects of the question and providing comprehensive information."""
}

language_criteria = {
    "language": """
Your task is to check if text1 and text2 are in the same language. If they are in the same language you should answer '1'. \
    If they are in the different languages, you should answer '0'.\
    After providing your explanation, you must rate the response either 0 or 1 \
by strictly following this format: '[[rating]]', for example: 'Rating: [[0]]
"""
}

complete_sentence_criteria = {
    "complete_sentence": "Evaluate if the last sentence of the text is a finished sentence with a period sign at the end of the last sentence. \
        If this it is a finished sentence score it as 1, if it is not, score it as 0.\
         After providing your explanation, you must rate the response either 0 or 1 \
by strictly following this format: '[[rating]]', for example: 'Rating: [[0]]"
}
