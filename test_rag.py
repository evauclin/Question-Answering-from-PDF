from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_cuisson_pates():
    assert query_and_validate(
        question="Comment bien cuire les pâtes ?",
        expected_response="Non presente dans le document",
    )


def test_resume_document():
    assert query_and_validate(
        question="Quel est le résumé de ce document ?",
        expected_response="C'est une introduction au Deep Learning, on parlera de fonction de cout, "
                          "resaux de neurones, etc.",
    )


def test_erreur_quadratique():
    assert query_and_validate(
        question="Que dit le document sur la fréquence d'utilisation de l’erreur quadratique moyenne ?",
        expected_response="L'erreur quadratique moyenne est la fonction de perte la plus couramment utilisée dans les"
                          " contextes de régression.",
    )


def test_fonction_entrainement():
    assert query_and_validate(
        question="Afin d’entraîner ces modèles d’ajuster leurs paramètres pour qu’ils s’adaptent aux données,"
                 " quelle fonction doit être définie ?",
        expected_response="Votre réponse ici",
    )


def test_derivative_relu():
     assert query_and_validate(
        question="Que peut-on constater avec la dérivée de ReLU ?",
        expected_response="On peut constater que la dérivée de ReLU possède une plus grande plage de valeurs d’entrée pour lesquelles "
                          "elle est non nulle (typiquement toute la plage de valeurs d’entrée positives) que ses concurrentes, "
                          "ce qui en fait une fonction d’activation très intéressante pour les réseaux neuronaux profonds",
    )


def test_early_stopping():
    assert query_and_validate(
        question="Expliquez le early stopping ?",
        expected_response="Le early stopping est une technique qui consiste à arrêter le processus d’apprentissage dès que la perte"
                          "de validation cesse de s’améliorer"
    )


def test_llm():
    assert query_and_validate(
        question="Explique ce qu'est un LLM ?",
        expected_response="Aucune information sur les LLM dans le document",
    )


def test_conv_nets():
    assert query_and_validate(
        question="Comment sont appelés les réseaux ConvNets ?",
        expected_response="Les convnets sont aussi neurones convolutifs",
    )


def test_perceptrons_multicouches():
    assert query_and_validate(
        question="Que dit le document sur les PERCEPTRONS MULTICOUCHES ?",
        expected_response="Un presentation des perceptrons multicouches est faite en mettant en valeur l'architecture de ces derniers"
                            "ainsi que leur fonctionnement",
    )


def test_theoreme_approximation():
    assert query_and_validate(
        question="Que stipule le théorème d’approximation universelle énoncé dans le document ?",
        expected_response="Le théorème d’approximation universelle stipule que toute fonction continue définie sur un ensemble compact peut être"
                          "approchée d’aussi près que l’on veut par un réseau neuronal à une couche cachée avec activation sigmoïde."
    )


def test_auteur_document():
    assert query_and_validate(
        question="Qui est l'auteur de ce document ?",
        expected_response="Romain Tavenard",
    )


def test_homer_simpson():
    assert query_and_validate(
        question="Qui est Homer Simpson ?",
        expected_response="Pas de réponse dans le document",
    )


def test_fonctions_activation():
    assert query_and_validate(
        question="Parle-t-on de fonction d’activation dans ce document ?",
        expected_response="Oui",
    )


def test_llm_document():
    assert query_and_validate(
        question="Parle-t-on de LLM dans ce document ?",
        expected_response="Non",
    )


def test_perte_courante():
    assert query_and_validate(
        question="Quelle est la perte la plus couramment utilisée ?",
        expected_response="L’erreur quadratique moyenne (ou Mean Squared Error, MSE) est la fonction de perte la plus"
                          "couramment utilisée dans les contextes de régression.",
    )


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
