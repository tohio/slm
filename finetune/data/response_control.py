"""
Generated response-control SFT examples.

These examples target small-model behavior weaknesses:
- direct arithmetic
- concise factual answers
- transformer / RNN distinction
- factual restraint for private or unverifiable claims
- clean stopping / exact-format answers

The goal is distinct examples, not repeated tiny handcrafted rows.
"""

from __future__ import annotations


def _record(system: str, user: str, assistant: str, sft_type: str) -> dict:
    return {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": "response_control",
        "sft_type": sft_type,
    }


def arithmetic_examples(system: str) -> list[dict]:
    records = []

    templates = [
        "What is {expr}?",
        "Answer only the result: {expr}",
        "Compute {expr}.",
        "What does {expr} equal?",
        "Give the answer to {expr}.",
        "Return only the number: {expr}",
    ]

    # Addition
    for a in range(1, 31):
        for b in range(1, 31):
            expr = f"{a} + {b}"
            answer = str(a + b)
            for template in templates[:2]:
                records.append(_record(system, template.format(expr=expr), answer, "arithmetic"))

    # Subtraction, non-negative
    for a in range(2, 41):
        for b in range(1, min(a, 20) + 1):
            expr = f"{a} - {b}"
            answer = str(a - b)
            records.append(_record(system, templates[(a + b) % len(templates)].format(expr=expr), answer, "arithmetic"))

    # Multiplication
    for a in range(2, 16):
        for b in range(2, 16):
            expr = f"{a} * {b}"
            answer = str(a * b)
            records.append(_record(system, templates[(a * b) % len(templates)].format(expr=expr), answer, "arithmetic"))

    # Exact division
    for divisor in range(2, 16):
        for quotient in range(1, 21):
            dividend = divisor * quotient
            expr = f"{dividend} / {divisor}"
            answer = str(quotient)
            records.append(_record(system, templates[(dividend + divisor) % len(templates)].format(expr=expr), answer, "arithmetic"))

    # Word-form anchors for known failure mode.
    word_examples = [
        ("What is two plus two?", "4"),
        ("Two plus two equals what?", "4"),
        ("Answer only the number: two plus two", "4"),
        ("What is three plus four?", "7"),
        ("What is ten minus six?", "4"),
        ("What is four times two?", "8"),
        ("What is eight divided by two?", "4"),
        ("What is twelve divided by three?", "4"),
    ]
    for user, answer in word_examples:
        records.append(_record(system, user, answer, "arithmetic"))

    return records


def simple_factual_examples(system: str) -> list[dict]:
    facts = [
        ("What is the capital of France?", "Paris."),
        ("What is the capital of Japan?", "Tokyo."),
        ("What is the capital of Italy?", "Rome."),
        ("What is the capital of Spain?", "Madrid."),
        ("What is the capital of Canada?", "Ottawa."),
        ("What is the capital of Germany?", "Berlin."),
        ("What is the capital of Brazil?", "Brasília."),
        ("What is the capital of India?", "New Delhi."),
        ("What color is the sky on a clear day?", "Blue."),
        ("What planet is known as the Red Planet?", "Mars."),
        ("What is the boiling point of water at sea level in Celsius?", "100°C."),
        ("What gas do humans need to breathe?", "Oxygen."),
        ("What is the largest ocean on Earth?", "The Pacific Ocean."),
        ("What is the smallest prime number?", "2."),
        ("How many days are in a normal week?", "7."),
    ]

    templates = [
        "{q}",
        "Answer briefly: {q}",
        "Give a concise answer: {q}",
        "Answer in one sentence: {q}",
        "Answer directly: {q}",
        "Give only the answer: {q}",
        "What is the correct answer? {q}",
        "Respond with the factual answer: {q}",
        "Keep it short: {q}",
        "Do not elaborate. {q}",
        "Give the answer without extra context: {q}",
        "Short answer only: {q}",
        "State the answer: {q}",
        "What is the answer to this? {q}",
        "Reply factually and stop: {q}",
        "Give a simple factual answer: {q}",
        "Answer with the right level of detail: {q}",
        "No explanation needed: {q}",
        "One short answer: {q}",
        "Just answer: {q}",
    ]

    records = []
    for q, a in facts:
        for template in templates:
            records.append(_record(system, template.format(q=q), a, "simple_factual"))

    return records

def ai_concept_examples(system: str) -> list[dict]:
    pairs = [
        (
            "In AI, what is a transformer model?",
            "A transformer is a neural network architecture that uses attention mechanisms to process sequences and learn relationships between tokens.",
        ),
        (
            "What is a transformer model in machine learning?",
            "A transformer is a neural network architecture based on attention, commonly used for language and sequence modeling.",
        ),
        (
            "Is a transformer the same as an RNN?",
            "No. A transformer uses attention mechanisms, while an RNN processes sequences recurrently.",
        ),
        (
            "Does a transformer mainly rely on recurrence?",
            "No. Transformers mainly rely on attention mechanisms, not recurrence.",
        ),
        (
            "What mechanism is central to transformer models?",
            "Attention is central to transformer models.",
        ),
        (
            "What is self-attention in a transformer?",
            "Self-attention lets each token attend to other tokens in the sequence to build contextual representations.",
        ),
        (
            "Compare transformers and RNNs briefly.",
            "Transformers use attention to model token relationships, while RNNs process sequences recurrently through hidden states.",
        ),
        (
            "Why are transformers useful for language models?",
            "Transformers are useful because attention lets them model relationships between tokens across a sequence efficiently.",
        ),
        (
            "What is a neural network?",
            "A neural network is a machine learning model made of connected layers that learn patterns from data.",
        ),
        (
            "What is next-token prediction?",
            "Next-token prediction trains a language model to predict the next token from the previous tokens.",
        ),
        (
            "What is an embedding in machine learning?",
            "An embedding is a learned vector representation of an item such as a token, word, or user.",
        ),
        (
            "What is overfitting?",
            "Overfitting happens when a model memorizes training data too closely and performs poorly on new data.",
        ),
    ]

    templates = [
        "{q}",
        "Answer clearly: {q}",
        "Briefly answer: {q}",
        "In one or two sentences, answer: {q}",
        "Give a concise ML answer: {q}",
        "Explain directly: {q}",
        "Give the correct machine-learning definition: {q}",
        "Answer without extra background: {q}",
        "State the concept clearly: {q}",
        "Give a short technical answer: {q}",
        "Answer for a beginner: {q}",
        "Answer factually: {q}",
        "Keep it concise: {q}",
        "Define this clearly: {q}",
        "What is the correct explanation? {q}",
        "Give the key idea: {q}",
        "Answer in plain language: {q}",
        "Give a precise answer: {q}",
        "Do not confuse related architectures. {q}",
        "Avoid incorrect RNN comparisons. {q}",
        "Use attention in the explanation. {q}",
        "Mention the central mechanism. {q}",
        "Give a sequence-modeling answer: {q}",
        "Answer as an ML tutor: {q}",
        "Answer and stop: {q}",
    ]

    records = []
    for q, a in pairs:
        for template in templates:
            records.append(_record(system, template.format(q=q), a, "ai_concept"))

    return records

def factual_restraint_examples(system: str) -> list[dict]:
    prompts = [
        "What was Anthropic's private revenue last month?",
        "What was OpenAI's exact internal revenue yesterday?",
        "What is the private salary of Anthropic's CEO this month?",
        "What is Apple's current private unreleased product roadmap?",
        "Give me the exact current bank balance of Microsoft.",
        "What is my neighbor's private phone number?",
        "What is the private home address of a random employee at Google?",
        "What private acquisition is Apple planning next week?",
        "What is the exact unreleased revenue number for a private startup today?",
        "What is the current private payroll amount for Anthropic?",
        "What is the exact private valuation of a startup that has not announced it?",
        "What is the exact internal headcount of OpenAI today?",
    ]

    answers = [
        "I do not have enough verified public information to answer that.",
        "I do not have access to verified private information about that.",
        "I cannot verify that private or unreleased information.",
        "I do not have verified public information for that specific private detail.",
    ]

    templates = [
        "{q}",
        "Answer carefully: {q}",
        "Be factual and avoid guessing: {q}",
        "If you cannot verify it, say so. {q}",
        "Do not invent private details. {q}",
        "Give a safe factual answer: {q}",
        "Answer only if verified: {q}",
        "What is the verified answer? {q}",
        "Avoid hallucinated specifics: {q}",
        "Use factual restraint: {q}",
        "Do not make up numbers. {q}",
        "If the information is private, say you cannot verify it. {q}",
        "Answer honestly: {q}",
        "Be careful with private/current claims: {q}",
        "State whether you can verify this: {q}",
        "Give the correct uncertainty response: {q}",
        "Do not claim access to private data. {q}",
        "Respond without unsupported details: {q}",
        "No speculation: {q}",
        "Answer and stop: {q}",
    ]

    records = []
    for i, prompt in enumerate(prompts):
        for j, template in enumerate(templates):
            answer = answers[(i + j) % len(answers)]
            records.append(_record(system, template.format(q=prompt), answer, "factual_restraint"))

    uncertainty_prompts = [
        "What should you do when you are not sure about a factual claim?",
        "How should you answer if a fact is private or not publicly verified?",
        "Should you invent a specific number if you cannot verify it?",
        "What should you say when current private information is unavailable?",
    ]
    uncertainty_answers = [
        "I should say that I am not sure or cannot verify it, rather than inventing details.",
        "I should clearly state that I cannot verify the private or non-public information.",
        "No. I should not invent specific numbers when I cannot verify them.",
        "I should say that I do not have enough verified information to answer.",
    ]

    for prompt, answer in zip(uncertainty_prompts, uncertainty_answers):
        for template in templates[:10]:
            records.append(_record(system, template.format(q=prompt), answer, "factual_restraint"))

    return records

def concise_answer_examples(system: str) -> list[dict]:
    base = [
        ("Say only: hello", "hello"),
        ("Return exactly one word: done", "done"),
        ("Answer with only the number: 3", "3"),
        ("Answer only the result: 10 / 2", "5"),
        ("Answer only the country: What country is Paris in?", "France"),
        ("Answer only the city: What is the capital of Japan?", "Tokyo"),
        ("Give only yes or no: Is Python a programming language?", "Yes."),
        ("Give only yes or no: Is the moon a star?", "No."),
        ("Answer in one sentence: What is Python?", "Python is a high-level programming language."),
        ("Answer in one sentence: What is a database?", "A database is an organized system for storing and retrieving data."),
    ]

    templates = [
        "{q}",
        "Follow the requested format exactly. {q}",
        "Do not add extra explanation. {q}",
        "Answer and stop. {q}",
        "Keep the answer minimal. {q}",
        "Return only what was requested. {q}",
        "No extra words. {q}",
        "Be concise. {q}",
        "Use the shortest correct answer. {q}",
        "Do not elaborate. {q}",
    ]

    records = []
    for q, a in base:
        for template in templates:
            records.append(_record(system, template.format(q=q), a, "concise_answer"))

    return records

def build_response_control_records(
    system: str,
    max_examples: int = 2000,
) -> list[dict]:
    """Return a balanced response-control SFT mix.

    Do not take a naive prefix of the generated records. Arithmetic produces
    many examples, so prefix slicing would drop every other behavior category.
    """
    buckets = {
        "arithmetic": arithmetic_examples(system),
        "simple_factual": simple_factual_examples(system),
        "ai_concept": ai_concept_examples(system),
        "factual_restraint": factual_restraint_examples(system),
        "concise_answer": concise_answer_examples(system),
    }

    # Conservative 125M mix. Arithmetic gets the largest share because it is
    # the most brittle observed failure, but factual restraint and AI concept
    # grounding must remain visible.
    target_mix = {
        "arithmetic": 900,
        "simple_factual": 250,
        "ai_concept": 250,
        "factual_restraint": 250,
        "concise_answer": 100,
    }

    records = []
    for sft_type, target in target_mix.items():
        records.extend(buckets[sft_type][:target])

    # Add any remaining examples round-robin until max_examples is reached.
    positions = {key: min(target_mix.get(key, 0), len(value)) for key, value in buckets.items()}
    keys = list(buckets)

    while len(records) < max_examples:
        added = False
        for key in keys:
            pos = positions[key]
            if pos < len(buckets[key]):
                records.append(buckets[key][pos])
                positions[key] += 1
                added = True
                if len(records) >= max_examples:
                    break
        if not added:
            break

    return records[:max_examples]
