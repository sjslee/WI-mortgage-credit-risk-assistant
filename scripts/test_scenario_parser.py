from src.scenario_parser import parse_scenario_intent


questions = [
    "What if the credit score increased to 760?",
    "What credit score gets PD below 0.50%?",
    "What DTI would move the borrower into the Low risk tier?",
    "Compare credit score 760 and DTI 30.",
    "Which variable produces the largest PD reduction?",
]


for question in questions:
    result = parse_scenario_intent(question)

    print()
    print("QUESTION:", question)
    print("INTENT:", result)