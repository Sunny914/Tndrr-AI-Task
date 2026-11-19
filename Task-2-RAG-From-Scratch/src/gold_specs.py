"""Candidate evaluation questions grounded in NIST AI RMF 1.0.

Needles are distinctive phrases that must occur on the listed physical PDF pages.
human_verified is never set here; only a human review step may mark questions verified.
"""

QUESTION_SPECS = [
    {
        "id": "Q01",
        "question": "What are the four core functions of the AI RMF?",
        "expected_answer": "GOVERN, MAP, MEASURE, and MANAGE.",
        "gold_pages": [25],
        "needle": "the Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE",
        "notes": "Section 5, AI RMF Core.",
    },
    {
        "id": "Q02",
        "question": "How does the AI RMF define an AI system?",
        "expected_answer": (
            "An engineered or machine-based system that can, for a given set of objectives, "
            "generate outputs such as predictions, recommendations, or decisions influencing "
            "real or virtual environments, designed to operate with varying levels of autonomy."
        ),
        "gold_pages": [6],
        "needle": "an engineered or machine-based system that can, for a given set of objectives",
        "notes": "Executive Summary definition adapted from OECD/ISO.",
    },
    {
        "id": "Q03",
        "question": "In the context of the AI RMF, what does risk refer to?",
        "expected_answer": (
            "The composite measure of an event's probability of occurring and the magnitude "
            "or degree of the consequences of the corresponding event."
        ),
        "gold_pages": [9],
        "needle": "risk refers to the composite measure of an event's probability of occurring",
        "notes": "Section 1.1.",
    },
    {
        "id": "Q04",
        "question": "Which characteristics of trustworthy AI systems does the Framework list?",
        "expected_answer": (
            "Valid and reliable, safe, secure and resilient, accountable and transparent, "
            "explainable and interpretable, privacy-enhanced, and fair with harmful bias managed."
        ),
        "gold_pages": [17],
        "needle": "valid and reliable, safe, secure and resilient, accountable and transparent",
        "notes": "Section 3 opening list.",
    },
    {
        "id": "Q05",
        "question": "Does the AI RMF prescribe risk tolerance for organizations?",
        "expected_answer": "No. It can be used to prioritize risk, but it does not prescribe risk tolerance.",
        "gold_pages": [12],
        "needle": "While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance",
        "notes": "Section 1.2.2.",
    },
    {
        "id": "Q06",
        "question": (
            "What should happen if an AI system presents unacceptable negative risk levels, "
            "such as imminent significant negative impacts or catastrophic risks?"
        ),
        "expected_answer": (
            "Development and deployment should cease in a safe manner until risks can be "
            "sufficiently managed."
        ),
        "gold_pages": [13],
        "needle": "development and deployment should cease in a safe manner until risks can be sufficiently managed",
        "notes": "Section 1.2.3.",
    },
    {
        "id": "Q07",
        "question": "How does the Framework define residual risk?",
        "expected_answer": "Risk remaining after risk treatment, citing ISO GUIDE 73.",
        "gold_pages": [13],
        "needle": "Residual risk – defined as risk remaining after risk treatment",
        "notes": "Section 1.2.3; en-dash in source.",
    },
    {
        "id": "Q08",
        "question": "What does TEVV stand for in the AI RMF?",
        "expected_answer": "Test, evaluation, verification, and validation.",
        "gold_pages": [14],
        "needle": "test, evaluation, verification, and validation (TEVV)",
        "notes": "Section 2 Audience.",
    },
    {
        "id": "Q09",
        "question": "How does the AI RMF distinguish explainability from interpretability?",
        "expected_answer": (
            "Explainability refers to a representation of the mechanisms underlying AI systems' "
            "operation, whereas interpretability refers to the meaning of AI systems' output in "
            "the context of their designed functional purposes."
        ),
        "gold_pages": [21],
        "needle": "Explainability refers to a representation of the mechanisms underlying AI systems' operation, whereas interpretability refers to the meaning of AI systems' output",
        "notes": "Section 3.5.",
    },
    {
        "id": "Q10",
        "question": "What three major categories of AI bias has NIST identified?",
        "expected_answer": "Systemic, computational and statistical, and human-cognitive.",
        "gold_pages": [23],
        "needle": "three major categories of AI bias to be considered and managed: systemic, computational and statistical, and human-cognitive",
        "notes": "Section 3.7.",
    },
    {
        "id": "Q11",
        "question": "Why is GOVERN described as a cross-cutting function?",
        "expected_answer": (
            "GOVERN is infused throughout AI risk management and enables the other functions "
            "of the process."
        ),
        "gold_pages": [27],
        "needle": "GOVERN is a cross-cutting function that is infused throughout AI risk management",
        "notes": "Section 5.1.",
    },
    {
        "id": "Q12",
        "question": (
            "After completing the MAP function, what initial decision should Framework users "
            "have enough contextual knowledge to make?"
        ),
        "expected_answer": (
            "An initial go/no-go decision about whether to design, develop, or deploy an AI system."
        ),
        "gold_pages": [30],
        "needle": "initial go/no-go decision about whether to design, develop, or deploy an AI system",
        "notes": "Section 5.2.",
    },
    {
        "id": "Q13",
        "question": "What is an AI RMF use-case profile?",
        "expected_answer": (
            "An implementation of the AI RMF functions, categories, and subcategories for a "
            "specific setting or application based on the user's requirements, risk tolerance, "
            "and resources; for example, a hiring profile or a fair housing profile."
        ),
        "gold_pages": [38],
        "needle": "AI RMF use-case profiles are implementations of the AI RMF functions, categories, and subcategories",
        "notes": "Section 6.",
    },
    {
        "id": "Q14",
        "question": (
            "According to Appendix B, what privacy-related AI-specific risk is listed as new "
            "or increased compared with traditional software?"
        ),
        "expected_answer": "Privacy risk due to enhanced data aggregation capability for AI systems.",
        "gold_pages": [43],
        "needle": "Privacy risk due to enhanced data aggregation capability for AI systems",
        "notes": "Appendix B.",
    },
    {
        "id": "Q15",
        "question": "What is the first listed development attribute that the AI RMF strives to satisfy?",
        "expected_answer": "Be risk-based, resource-efficient, pro-innovation, and voluntary.",
        "gold_pages": [47],
        "needle": "Be risk-based, resource-efficient, pro-innovation, and voluntary",
        "notes": "Appendix D.",
    },
]
